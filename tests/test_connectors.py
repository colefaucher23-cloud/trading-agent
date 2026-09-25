import pandas as pd
import pytest

from kvwap import connectors
from kvwap.data import bars_to_grid


def test_alpaca_parsing_and_pagination(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    pages = [
        {"bars": [{"t": "2026-03-02T14:30:00Z", "o": 1, "h": 1, "l": 1, "c": 10.0, "v": 100, "vw": 10.1}],
         "next_page_token": "p2"},
        {"bars": [{"t": "2026-03-02T14:45:00Z", "o": 1, "h": 1, "l": 1, "c": 11.0, "v": 200, "vw": 10.9}],
         "next_page_token": None},
    ]
    calls = []

    def fake_get(url, params, headers=None, retries=4):
        calls.append(dict(params))
        assert headers["APCA-API-KEY-ID"] == "k"
        return pages[len(calls) - 1]

    monkeypatch.setattr(connectors, "_http_get", fake_get)
    src = connectors.AlpacaDataSource(feed="sip")
    assert src.delay_minutes == 16
    df = src.get_bars("SPY", pd.Timestamp("2026-03-01", tz="UTC"), pd.Timestamp("2026-03-03", tz="UTC"))
    assert list(df.columns) == ["close", "volume", "vwap"]
    assert len(df) == 2 and calls[1]["page_token"] == "p2"
    assert calls[0]["feed"] == "sip" and calls[0]["adjustment"] == "split"


def test_alphavantage_parsing(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "demo")
    # end-labelled bars 09:45 .. 16:00 for one full day
    idx = pd.date_range("2026-03-02 09:45", "2026-03-02 16:00", freq="15min")
    series = {ts.strftime("%Y-%m-%d %H:%M:%S"): {"4. close": "100.0", "5. volume": "1000"} for ts in idx}
    monkeypatch.setattr(connectors, "_http_get", lambda url, params, headers=None, retries=4: {"Time Series (15min)": series})
    monkeypatch.setattr(connectors.time, "sleep", lambda s: None)
    src = connectors.AlphaVantageSource()
    df = src.get_bars("IBM", pd.Timestamp("2026-03-01", tz="UTC"), pd.Timestamp("2026-03-03", tz="UTC"))
    assert len(df) == 26
    g = bars_to_grid(df)
    assert g.volume.shape == (1, 26)


def test_alphavantage_premium_message_raises(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "demo")
    monkeypatch.setattr(connectors, "_http_get", lambda *a, **k: {"Information": "premium endpoint"})
    with pytest.raises(RuntimeError, match="premium"):
        connectors.AlphaVantageSource().get_bars("IBM", pd.Timestamp("2026-03-01", tz="UTC"), pd.Timestamp("2026-03-02", tz="UTC"))


def test_missing_keys_raise(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="APCA_API_KEY_ID"):
        connectors.AlpacaDataSource()


def test_live_broker_needs_explicit_opt_in(monkeypatch):
    from kvwap.broker import AlpacaBroker

    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    monkeypatch.delenv("KVWAP_ALLOW_LIVE", raising=False)
    with pytest.raises(RuntimeError, match="KVWAP_ALLOW_LIVE"):
        AlpacaBroker(live=True)
    assert AlpacaBroker().base == AlpacaBroker.PAPER
