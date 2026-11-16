# BTC Options Straddle Research Dashboard

Streamlit workspace for live BTC candles and Deribit option quotes, hybrid BTC forecasting research, dynamic same-strike pair analysis, local Ollama chat, and explicitly controlled Deribit TestNet workflows. Production market data is public and read-only. TestNet authentication and orders require user action; order submission is disabled by default.

## Start the application

Use 64-bit CPython 3.12.10. From PowerShell at the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m streamlit run streamlit_app.py
```

`requirements.txt` is the only dependency file. It contains the Streamlit app, exchange runtime, market-data, feature pipeline, and CPU model dependencies. Install it once in the same environment used to launch Streamlit. The app entrypoint, `.streamlit/config.toml`, and Python packages intentionally remain at the repository root so Streamlit imports work from the project directory.

Ollama is optional and runs locally. Start the Ollama service, install a model, then use **Refresh local models** in BTC Chat. Prompts and the dashboard snapshot are sent to the local Ollama endpoint at `127.0.0.1:11434`.

## Streamlit sections

- **Live Deribit:** public live BTC candles, Deribit option quotes, selected-pair risk, and AI research.
- **BTC Chat:** local Ollama chat grounded in the current dashboard snapshot.
- **Deribit TestNet:** separate TestNet instruments/account controls and explicitly confirmed order workflow.

Configure TestNet credentials only in the local `.env`, using a dedicated TestNet API key with `account:read` and `trade:read_write` scopes. `DERIBIT_TESTNET_ORDER_SUBMISSION_ENABLED` defaults to `false`; enable it only when intentionally testing TestNet orders, then restart Streamlit. The authenticated client is fixed to `https://test.deribit.com/api/v2`. Production order submission is not implemented. Never commit `.env`.

## Project layout

```text
BTC_Options_Straddle/
├── streamlit_app.py          # Streamlit application entrypoint
├── .streamlit/config.toml     # App theme and appearance
├── requirements.txt          # Single pinned dependency set
├── .env.example              # Safe environment-variable template
├── config/                   # Runtime, risk, and settings code
├── configs/                  # YAML-compatible strategy settings
├── data/                     # Market data and feature pipeline
├── models/                   # Forecasts, scenarios, model training
├── options/                  # Option chain and straddle analysis
├── strategies/               # Dynamic selection and risk filters
├── execution/                # Paper accounting and TestNet services
├── backtesting/              # Historical research and metrics
├── chat/                     # Local Ollama client
├── scripts/                  # Data, training, and analysis commands
├── tests/                    # Automated checks
├── docs/                     # Guides, audits, and project report
├── deployment_assets/        # Curated inference bundle and 365-day scenario history
├── runtime.py                # Shared app runtime services
└── main.py                   # Legacy CLI runtime entrypoint
```

Raw downloads, full processed datasets, reports, local wheels, runtime state, and secrets are excluded from Git. The small, curated `deployment_assets/` inference bundle and 365-day public BTC scenario-history snapshot are versioned so Streamlit deployments can run model-backed risk analysis. Rebuild this bundle after promoting new model weights or refreshing the research history:

```powershell
python -m scripts.package_streamlit_assets
```

The app prefers full local data under `data/processed/` and falls back to `deployment_assets/` when running from a deployment checkout.

## Research workflow

After placing candle CSV files under `data/raw/`, run:

```powershell
python -m scripts.inspect_datasets
python -m scripts.preprocess_datasets
python -m scripts.build_features
python -m scripts.prepare_training_data
python -m scripts.train_models
```

The hybrid model combines LSTM and XGBoost research forecasts. Training writes an isolated timestamped experiment by default. Promotion to `data/processed/model_artifacts/` must be explicit. To generate a prepared-data prediction or reliability report:

```powershell
python -m scripts.predict_hybrid
python -m scripts.evaluate_model_reliability
```

In Live Deribit → AI research, the app combines completed candle features, forecast/scenario inputs, and fresh option quotes to rank same-expiry, same-strike Call–Put pairs. It reports expected P&L and risk estimates and can return **NO_TRADE** if required inputs, liquidity, or configured risk limits fail. Candle patterns are descriptive risk context; scenario losses are estimates, not maximum-loss bounds. The model has not established profitable performance. See `docs/guides/` for the detailed model and system guides.

For walk-forward evaluation and a mark-price proxy comparison:

```powershell
python -m scripts.evaluate_walk_forward --features data/processed/btc_multitimeframe_features.parquet --max-folds 3
python -m scripts.backtest_mark_proxy
```

The available historical option tape lacks bid/ask depth and verified fills, so mark-price results are not an executable-price backtest. Read the generated reports under `data/reports/` and the audits in `docs/audits/` before interpreting results as evidence of strategy performance.

## Safety and execution limits

The two TestNet entry legs are separate exchange orders, not an atomic trade. Entry orders are post-only; risk and recovery exits use market orders and can incur slippage. Automation pauses for manual review when order state, position reconciliation, cancellation, or close execution cannot be verified. TestNet operation is for synthetic-fund testing and is not a production execution guarantee.
