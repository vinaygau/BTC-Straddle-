from nautilus_trader.adapters.deribit.config import DeribitEnvironment

from config import build_deribit_client_configs, get_deribit_connection_status


def test_deribit_config_builder_uses_testnet(monkeypatch):
    monkeypatch.setenv("DERIBIT_TESTNET_CLIENT_ID", "demo_client_id")
    monkeypatch.setenv("DERIBIT_TESTNET_CLIENT_SECRET", "demo_client_secret")
    monkeypatch.setenv("DERIBIT_TESTNET", "true")

    data_cfg, exec_cfg = build_deribit_client_configs()

    assert data_cfg.environment == DeribitEnvironment.TESTNET
    assert exec_cfg.environment == DeribitEnvironment.TESTNET
    assert data_cfg.api_key == "demo_client_id"
    assert exec_cfg.api_key == "demo_client_id"


def test_deribit_connection_status_reports_readiness(monkeypatch):
    monkeypatch.setenv("DERIBIT_TESTNET", "true")
    monkeypatch.setenv("DERIBIT_TESTNET_CLIENT_ID", "demo_client_id")
    monkeypatch.setenv("DERIBIT_TESTNET_CLIENT_SECRET", "demo_client_secret")

    status = get_deribit_connection_status()
    assert status["deribit_testnet"] is True
    assert status["credentials_loaded"] is True
    assert status["environment"] == "TESTNET"
