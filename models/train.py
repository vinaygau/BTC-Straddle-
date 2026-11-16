"""Offline-only LSTM, XGBoost, and validation-weighted hybrid training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import chronological_splits, fit_feature_scaler, prepare_training_frame, transform_features
from .training_config import ModelTrainingConfig


def _regression_metrics(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}


def _classification_metrics(actual, probabilities):
    probabilities = np.asarray(probabilities)
    actual = np.asarray(actual, dtype=int)
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": float(np.mean(predicted == actual)),
        "log_loss": float(-np.log(np.clip(probabilities[np.arange(len(actual)), actual], 1e-12, 1)).mean()),
    }


def _sequence_batch(indices, values, targets, timestamps, sequence_length, sequence_stride=1):
    xs, kept = [], []
    cadence = pd.Timedelta(minutes=5)
    if sequence_stride <= 0:
        raise ValueError("sequence_stride must be positive")
    for offset, end in enumerate(indices):
        if offset % sequence_stride:
            continue
        start = end - sequence_length + 1
        if start < 0: continue
        window_times = timestamps.iloc[start:end + 1]
        if len(window_times) != sequence_length or not window_times.diff().dropna().eq(cadence).all(): continue
        xs.append(values[start:end + 1])
        kept.append(end)
    if not kept:
        raise ValueError("No contiguous sequences available for the requested split")
    kept = np.asarray(kept, dtype=int)
    return np.asarray(xs, dtype=np.float32), kept, {key: np.asarray(value)[kept] for key, value in targets.items()}


def _build_torch_network(torch, input_size):
    from torch import nn

    class MultiTaskLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.recurrent = nn.LSTM(input_size, 64, num_layers=2, batch_first=True, dropout=0.15)
            self.shared = nn.Sequential(nn.Linear(64, 48), nn.ReLU(), nn.Dropout(0.1))
            self.return_quantiles = nn.Linear(48, 3)
            self.volatility = nn.Linear(48, 1)
            self.direction = nn.Linear(48, 3)
            self.extreme = nn.Linear(48, 1)

        def forward(self, values):
            output, _ = self.recurrent(values)
            shared = self.shared(output[:, -1, :])
            return (self.return_quantiles(shared), self.volatility(shared).squeeze(-1),
                    self.direction(shared), self.extreme(shared).squeeze(-1))

    return MultiTaskLSTM()


def _pinball_loss(torch, prediction, target, quantiles=(0.1, 0.5, 0.9)):
    losses = []
    for index, quantile in enumerate(quantiles):
        error = target - prediction[:, index]
        losses.append(torch.maximum(quantile * error, (quantile - 1.0) * error).mean())
    return sum(losses) / len(losses)


def train_models(feature_path: str | Path, output_dir: str | Path,
                 config: ModelTrainingConfig | None = None, *,
                 include_xgboost: bool = True) -> dict:
    """Train offline research models; no predictions are passed to strategy code."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "Model training requires optional packages. Install with: "
            "python -m pip install -r requirements-ml.txt"
        ) from exc
    if include_xgboost:
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost is unavailable. Install it from requirements-ml.txt or "
                "rerun with --lstm-only for an explicitly LSTM-only research pass."
            ) from exc

    config = config or ModelTrainingConfig()
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_features = pd.read_parquet(feature_path)
    labeled, feature_columns = prepare_training_frame(raw_features, config)
    splits = chronological_splits(labeled, feature_columns, config)
    train_rows, validation_rows, test_rows = splits.train, splits.validation, splits.test
    mean, scale = fit_feature_scaler(labeled, train_rows, feature_columns)
    np.savez(
        output_dir / "feature_scaler.npz",
        mean=mean,
        scale=scale,
        feature_columns=np.asarray(feature_columns),
    )
    values = transform_features(labeled, feature_columns, mean, scale)
    targets = {
        "return": labeled.target_return.to_numpy(dtype=np.float32),
        "volatility": labeled.target_volatility.to_numpy(dtype=np.float32),
        "direction": labeled.target_direction.to_numpy(dtype=np.int64),
        "extreme": labeled.target_extreme_move.to_numpy(dtype=np.int64),
    }
    return_mean = float(targets["return"][train_rows].mean())
    return_scale = float(targets["return"][train_rows].std())
    volatility_mean = float(targets["volatility"][train_rows].mean())
    volatility_scale = float(targets["volatility"][train_rows].std())
    return_scale = return_scale if np.isfinite(return_scale) and return_scale > 0 else 1.0
    volatility_scale = volatility_scale if np.isfinite(volatility_scale) and volatility_scale > 0 else 1.0
    xgb_train_rows = train_rows[::config.sequence_stride]
    x_train, x_val, x_test = values[xgb_train_rows], values[validation_rows], values[test_rows]
    report = {
        "scope": "research only; forecasts support advisory option ranking and are not submitted as orders",
        "feature_path": str(feature_path), "feature_count": len(feature_columns),
        "feature_columns": list(feature_columns), "horizon_steps": config.horizon_steps,
        "sequence_length": config.sequence_length, "sequence_stride": config.sequence_stride,
        "model_mode": "hybrid" if include_xgboost else "lstm_only",
        "training_config": {
            "random_seed": config.random_seed,
            "lstm_epochs_max": config.lstm_epochs,
            "lstm_batch_size": config.lstm_batch_size,
            "xgboost_estimators": config.xgboost_estimators if include_xgboost else None,
            "sequence_stride": config.sequence_stride,
        },
        "target_scaling": {
            "return_mean": return_mean, "return_scale": return_scale,
            "volatility_mean": volatility_mean, "volatility_scale": volatility_scale,
            "fit_split": "train",
        },
        "direction_threshold": config.direction_threshold,
        "extreme_move_threshold": config.extreme_move_threshold,
        "splits": {
            "train": {"rows": len(train_rows), "first": labeled.prediction_time_utc.iloc[train_rows[0]].isoformat(), "last": labeled.prediction_time_utc.iloc[train_rows[-1]].isoformat()},
            "validation": {"rows": len(validation_rows), "first": labeled.prediction_time_utc.iloc[validation_rows[0]].isoformat(), "last": labeled.prediction_time_utc.iloc[validation_rows[-1]].isoformat()},
            "test": {"rows": len(test_rows), "first": labeled.prediction_time_utc.iloc[test_rows[0]].isoformat(), "last": labeled.prediction_time_utc.iloc[test_rows[-1]].isoformat()},
        },
        "xgboost_training_rows": int(len(xgb_train_rows)),
    }

    xgb_validation = xgb_test = None
    if include_xgboost:
        tree_config = dict(n_estimators=config.xgboost_estimators, max_depth=4, learning_rate=0.04,
                           subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
                           tree_method="hist", n_jobs=4, random_state=config.random_seed)
        xgb_models = {
            "return": XGBRegressor(objective="reg:squarederror", **tree_config),
            "volatility": XGBRegressor(objective="reg:squarederror", **tree_config),
            "direction": XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", **tree_config),
            "extreme": XGBClassifier(objective="binary:logistic", eval_metric="logloss", **tree_config),
        }
        y = {key: targets[key] for key in targets}
        for name, model in xgb_models.items(): model.fit(x_train, y[name][xgb_train_rows], verbose=False)
        xgb_validation = {
            "return": xgb_models["return"].predict(x_val),
            "volatility": xgb_models["volatility"].predict(x_val),
            "direction": xgb_models["direction"].predict_proba(x_val),
            "extreme": xgb_models["extreme"].predict_proba(x_val)[:, 1],
        }
        xgb_test = {
            "return": xgb_models["return"].predict(x_test),
            "volatility": xgb_models["volatility"].predict(x_test),
            "direction": xgb_models["direction"].predict_proba(x_test),
            "extreme": xgb_models["extreme"].predict_proba(x_test)[:, 1],
        }
        for name, model in xgb_models.items(): model.save_model(output_dir / f"xgboost_{name}.json")

    timestamp_series = pd.to_datetime(labeled.prediction_time_utc, utc=True)
    sequence_sets = {}
    for split_name, indices in (("train", train_rows), ("validation", validation_rows), ("test", test_rows)):
        sequence_sets[split_name] = _sequence_batch(
            indices, values, targets, timestamp_series, config.sequence_length,
            sequence_stride=config.sequence_stride,
        )
    train_x, train_kept, train_y = sequence_sets["train"]
    val_x, val_kept, val_y = sequence_sets["validation"]
    test_x, test_kept, test_y = sequence_sets["test"]
    direction_prior = np.bincount(train_y["direction"], minlength=3).astype(np.float64)
    direction_prior /= direction_prior.sum()
    extreme_prior = float(train_y["extreme"].mean())
    report["baseline_metrics"] = {
        "zero_return": _regression_metrics(test_y["return"], np.zeros_like(test_y["return"])),
        "train_mean_return": _regression_metrics(
            test_y["return"], np.full_like(test_y["return"], train_y["return"].mean())
        ),
        "train_mean_volatility": _regression_metrics(
            test_y["volatility"], np.full_like(test_y["volatility"], train_y["volatility"].mean())
        ),
        "train_class_prior_direction": _classification_metrics(
            test_y["direction"], np.tile(direction_prior, (len(test_y["direction"]), 1))
        ),
        "train_class_prior_extreme": _classification_metrics(
            test_y["extreme"],
            np.tile([1.0 - extreme_prior, extreme_prior], (len(test_y["extreme"]), 1)),
        ),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = _build_torch_network(torch, values.shape[1]).to(device)
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y["return"]),
                               torch.from_numpy(train_y["volatility"]), torch.from_numpy(train_y["direction"]),
                               torch.from_numpy(train_y["extreme"]))
    loader = DataLoader(train_data, batch_size=config.lstm_batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=1e-4)
    best_val_loss, best_state, patience = float("inf"), None, 0
    for _epoch in range(config.lstm_epochs):
        network.train()
        for bx, by_ret, by_vol, by_dir, by_ext in loader:
            bx, by_ret, by_vol = bx.to(device), by_ret.to(device), by_vol.to(device)
            by_dir, by_ext = by_dir.to(device), by_ext.float().to(device)
            by_ret = (by_ret - return_mean) / return_scale
            by_vol = (by_vol - volatility_mean) / volatility_scale
            qret, volatility, direction, extreme = network(bx)
            loss = (_pinball_loss(torch, qret, by_ret) + nn.functional.smooth_l1_loss(volatility, by_vol)
                    + nn.functional.cross_entropy(direction, by_dir)
                    + nn.functional.binary_cross_entropy_with_logits(extreme, by_ext))
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(network.parameters(), 1.0); optimizer.step()
        network.eval()
        with torch.no_grad():
            vx = torch.from_numpy(val_x).to(device)
            vret, vvol, vdir, vext = network(vx)
            val_return = (torch.from_numpy(val_y["return"]).to(device) - return_mean) / return_scale
            val_volatility = (torch.from_numpy(val_y["volatility"]).to(device) - volatility_mean) / volatility_scale
            val_loss = float((_pinball_loss(torch, vret, val_return)
                              + nn.functional.smooth_l1_loss(vvol, val_volatility)
                              + nn.functional.cross_entropy(vdir, torch.from_numpy(val_y["direction"]).to(device))
                              + nn.functional.binary_cross_entropy_with_logits(vext, torch.from_numpy(val_y["extreme"].astype(np.float32)).to(device))).cpu())
        if val_loss < best_val_loss:
            best_val_loss, patience = val_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
        else:
            patience += 1
            if patience >= 4: break
    if best_state is None: raise RuntimeError("LSTM training did not produce a validation checkpoint")
    network.load_state_dict(best_state)
    torch.save({
        "state_dict": best_state,
        "input_size": values.shape[1],
        "sequence_length": config.sequence_length,
        "feature_columns": list(feature_columns),
        "return_mean": return_mean,
        "return_scale": return_scale,
        "volatility_mean": volatility_mean,
        "volatility_scale": volatility_scale,
    }, output_dir / "lstm_multitask.pt")

    def lstm_predict(batch):
        network.eval()
        with torch.no_grad():
            output = network(torch.from_numpy(batch).to(device))
            quantiles = output[0].cpu().numpy() * return_scale + return_mean
            volatility = np.maximum(0.0, output[1].cpu().numpy() * volatility_scale + volatility_mean)
            return {"quantiles": quantiles, "return": quantiles[:, 1],
                    "volatility": volatility, "direction": torch.softmax(output[2], dim=1).cpu().numpy(),
                    "extreme": torch.sigmoid(output[3]).cpu().numpy()}

    lstm_val, lstm_test = lstm_predict(val_x), lstm_predict(test_x)
    true_test_return = test_y["return"]
    test_predictions = pd.DataFrame({
        "prediction_time_utc": timestamp_series.iloc[test_kept].to_numpy(),
        "actual_return": true_test_return,
        "actual_direction": test_y["direction"],
        "actual_extreme_move": test_y["extreme"],
    })
    if include_xgboost:
        xgb_val_positions = np.searchsorted(validation_rows, val_kept)
        xgb_test_positions = np.searchsorted(test_rows, test_kept)
        # Weight the offline blend by validation return RMSE; held-out test rows are untouched.
        xgb_val_ret = xgb_validation["return"][xgb_val_positions]
        xgb_val_vol = xgb_validation["volatility"][xgb_val_positions]
        xgb_rmse = _regression_metrics(val_y["return"], xgb_val_ret)["rmse"]
        lstm_rmse = _regression_metrics(val_y["return"], lstm_val["return"])["rmse"]
        weights = np.array([1 / max(xgb_rmse, 1e-12), 1 / max(lstm_rmse, 1e-12)])
        weights /= weights.sum()
        xgb_te_ret = xgb_test["return"][xgb_test_positions]
        xgb_te_vol = xgb_test["volatility"][xgb_test_positions]
        hybrid_return = weights[0] * xgb_te_ret + weights[1] * lstm_test["return"]
        hybrid_volatility = np.maximum(0, weights[0] * xgb_te_vol + weights[1] * lstm_test["volatility"])
        hybrid_direction = (weights[0] * xgb_test["direction"][xgb_test_positions]
                            + weights[1] * lstm_test["direction"])
        hybrid_extreme = (weights[0] * xgb_test["extreme"][xgb_test_positions]
                          + weights[1] * lstm_test["extreme"])
        residuals = val_y["return"] - (weights[0] * xgb_val_ret + weights[1] * lstm_val["return"])
        residual_quantiles = np.quantile(residuals, [0.1, 0.5, 0.9]).tolist()
        calibrated_quantiles = hybrid_return[:, None] + np.asarray(residual_quantiles)[None, :]
        test_predictions["hybrid_expected_return"] = hybrid_return
        test_predictions["hybrid_return_q10"] = calibrated_quantiles[:, 0]
        test_predictions["hybrid_return_q50"] = calibrated_quantiles[:, 1]
        test_predictions["hybrid_return_q90"] = calibrated_quantiles[:, 2]
        test_predictions["hybrid_volatility"] = hybrid_volatility
        test_predictions["bearish_probability"] = hybrid_direction[:, 0]
        test_predictions["sideways_probability"] = hybrid_direction[:, 1]
        test_predictions["bullish_probability"] = hybrid_direction[:, 2]
        test_predictions["extreme_movement_probability"] = hybrid_extreme
        report["validation_blend_weights"] = {"xgboost": float(weights[0]), "lstm": float(weights[1])}
        report["return_quantile_residual_offsets"] = residual_quantiles
        report["test_metrics"] = {
            "xgboost_return": _regression_metrics(test_y["return"], xgb_te_ret),
            "lstm_return": _regression_metrics(test_y["return"], lstm_test["return"]),
            "hybrid_return": _regression_metrics(test_y["return"], hybrid_return),
            "hybrid_volatility": _regression_metrics(test_y["volatility"], hybrid_volatility),
            "hybrid_direction": _classification_metrics(test_y["direction"], hybrid_direction),
            "hybrid_extreme": _classification_metrics(test_y["extreme"], np.column_stack([1 - hybrid_extreme, hybrid_extreme])),
        }
    else:
        test_predictions["lstm_expected_return"] = lstm_test["return"]
        test_predictions["lstm_return_q10"] = lstm_test["quantiles"][:, 0]
        test_predictions["lstm_return_q50"] = lstm_test["quantiles"][:, 1]
        test_predictions["lstm_return_q90"] = lstm_test["quantiles"][:, 2]
        test_predictions["lstm_volatility"] = lstm_test["volatility"]
        test_predictions["bearish_probability"] = lstm_test["direction"][:, 0]
        test_predictions["sideways_probability"] = lstm_test["direction"][:, 1]
        test_predictions["bullish_probability"] = lstm_test["direction"][:, 2]
        test_predictions["extreme_movement_probability"] = lstm_test["extreme"]
        report["test_metrics"] = {
            "lstm_return": _regression_metrics(test_y["return"], lstm_test["return"]),
            "lstm_volatility": _regression_metrics(test_y["volatility"], lstm_test["volatility"]),
            "lstm_direction": _classification_metrics(test_y["direction"], lstm_test["direction"]),
            "lstm_extreme": _classification_metrics(test_y["extreme"], np.column_stack([1 - lstm_test["extreme"], lstm_test["extreme"]])),
        }
        report["validation_metrics"] = {
            "lstm_return": _regression_metrics(val_y["return"], lstm_val["return"]),
        }
    test_predictions.to_parquet(output_dir / "heldout_test_predictions.parquet", index=False)

    report["lstm_device"] = str(device)
    report["training_rows_with_sequences"] = len(train_kept)
    report["validation_rows_with_sequences"] = len(val_kept)
    report["test_rows_with_sequences"] = len(test_kept)
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

