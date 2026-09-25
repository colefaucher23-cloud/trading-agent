"""Free market-data connectors.

All sources return a DataFrame indexed by tz-aware bar time with at least
``close`` and ``volume`` columns (``vwap`` when the provider has it). Bars are
labelled by their start time, except Alpha Vantage, whose labelling
``bars_to_grid`` detects automatically.

| source        | cost | history at 15-min          | key needed                          |
|---------------|------|----------------------------|-------------------------------------|
| yahoo         | free | last ~60 days              | none                                |
| alpaca        | free | years (IEX or delayed SIP) | free paper account: APCA_API_KEY_ID  |
|               |      |                            | + APCA_API_SECRET_KEY               |
| alphavantage  | free | per month (free tier may   | ALPHAVANTAGE_API_KEY                |
|               |      | not include intraday)      |                                     |
| csv           | free | whatever you saved         | none                                |
"""
from __future__ import annotations

import os
import time
from typing import Callable, Optional, Protocol

import pandas as pd

from .data import load_csv


class BarSource(Protocol):
    def get_bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["close", "volume"], index=pd.DatetimeIndex([], tz="UTC"))


def _http_get(url: str, params: dict, headers: Optional[dict] = None, retries: int = 4) -> dict:
    import requests

    delay = 2.0
    for attempt in range(retries + 1):
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 429 and attempt < retries:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("unreachable")


class YahooSource:
    """Yahoo Finance via ``yfinance`` (pip install yfinance). No key, ~60 days of 15m bars."""

    def __init__(self, interval: str = "15m"):
        self.interval = interval

    def get_bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        import yfinance as yf

        start = max(_utc(start), pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=59))
        df = yf.Ticker(symbol).history(start=start, end=end, interval=self.interval, prepost=False, auto_adjust=False)
        if df.empty:
            return _empty()
        df = df.rename(columns=str.lower)[["close", "volume"]]
        return df


class AlpacaDataSource:
    """Alpaca Market Data API v2 (free with a paper-trading account).

    ``feed="iex"`` is real-time on the free plan but only covers IEX volume
    (a few % of the tape -- the intraday *shape* is similar). ``feed="sip"``
    is the consolidated tape; the free plan serves it with a 15-minute delay,
    which ``delay_minutes`` enforces. Train and trade on the same feed.
    """

    URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"

    def __init__(
        self,
        feed: str = "iex",
        timeframe: str = "15Min",
        key_id: Optional[str] = None,
        secret: Optional[str] = None,
        delay_minutes: Optional[int] = None,
    ):
        self.feed = feed
        self.timeframe = timeframe
        self.key_id = key_id or os.environ.get("APCA_API_KEY_ID")
        self.secret = secret or os.environ.get("APCA_API_SECRET_KEY")
        if not self.key_id or not self.secret:
            raise RuntimeError("set APCA_API_KEY_ID and APCA_API_SECRET_KEY (free at alpaca.markets)")
        self.delay_minutes = (16 if feed == "sip" else 0) if delay_minutes is None else delay_minutes

    def get_bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        end = min(_utc(end), pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=self.delay_minutes))
        params = {
            "timeframe": self.timeframe,
            "start": _utc(start).isoformat(),
            "end": end.isoformat(),
            "limit": 10000,
            "adjustment": "split",
            "feed": self.feed,
        }
        headers = {"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret}
        rows = []
        while True:
            data = _http_get(self.URL.format(symbol=symbol), params, headers)
            rows.extend(data.get("bars") or [])
            token = data.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        if not rows:
            return _empty()
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["t"], utc=True)
        return df.rename(columns={"c": "close", "v": "volume", "vw": "vwap"})[["close", "volume", "vwap"]]


class AlphaVantageSource:
    """Alpha Vantage TIME_SERIES_INTRADAY, fetched month by month.

    Get a free key at alphavantage.co. Some keys only get intraday data on a
    premium plan; the API then answers with an "Information" message, which is
    raised here as an error.
    """

    URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None, interval: str = "15min", tz: str = "America/New_York"):
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise RuntimeError("set ALPHAVANTAGE_API_KEY (free at alphavantage.co)")
        self.interval = interval
        self.tz = tz

    def get_bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        months = pd.period_range(_utc(start).tz_localize(None), _utc(end).tz_localize(None), freq="M")
        frames = []
        for m in months:
            data = _http_get(self.URL, {
                "function": "TIME_SERIES_INTRADAY", "symbol": symbol, "interval": self.interval,
                "month": str(m), "outputsize": "full", "extended_hours": "false",
                "adjusted": "true", "apikey": self.api_key,
            })
            key = f"Time Series ({self.interval})"
            if key not in data:
                raise RuntimeError(f"Alpha Vantage: {data.get('Information') or data.get('Note') or data.get('Error Message') or data}")
            df = pd.DataFrame.from_dict(data[key], orient="index")
            df.index = pd.to_datetime(df.index).tz_localize(self.tz)
            frames.append(df.rename(columns={"4. close": "close", "5. volume": "volume"})[["close", "volume"]].astype(float))
            time.sleep(1.0)
        if not frames:
            return _empty()
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated()]
        # Bar labelling (start vs end of bar) is auto-detected by bars_to_grid.
        return df[(df.index >= _utc(start)) & (df.index <= _utc(end))]


class CSVSource:
    """Bars previously saved with ``kvwap fetch`` (or any CSV with close/volume)."""

    def __init__(self, path: str):
        self.df = load_csv(path)

    def get_bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        idx = self.df.index
        return self.df[(idx >= _utc(start)) & (idx < _utc(end))]


class ReplaySource:
    """Serves historical bars as if live: only bars that have *finished* by
    ``clock()`` minus ``delay_minutes`` are visible. Used for simulation."""

    def __init__(self, bars: pd.DataFrame, clock: Callable[[], pd.Timestamp], bar_minutes: int = 15, delay_minutes: int = 0):
        df = bars.copy()
        idx = pd.DatetimeIndex(df.index)
        df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        self.df = df.sort_index()
        self.clock = clock
        self.bar = pd.Timedelta(minutes=bar_minutes)
        self.delay = pd.Timedelta(minutes=delay_minutes)

    def get_bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        visible_until = min(_utc(end), self.clock() - self.delay)
        idx = self.df.index
        return self.df[(idx >= _utc(start)) & (idx + self.bar <= visible_until)]

    def bar_price(self, ts: pd.Timestamp) -> float:
        """Price proxy for executing inside the bar containing ``ts``: the bar's
        VWAP if present, else its last price (the paper's proxy)."""
        idx = self.df.index
        row = self.df[(idx <= ts) & (idx + self.bar > ts)]
        if row.empty:
            row = self.df[idx <= ts].tail(1)
        r = row.iloc[-1]
        return float(r["vwap"]) if "vwap" in row and pd.notna(r.get("vwap")) else float(r["close"])


def make_source(name: str, **kw) -> BarSource:
    name = name.lower()
    if name == "yahoo":
        return YahooSource()
    if name == "alpaca":
        return AlpacaDataSource(feed=kw.get("feed", "iex"))
    if name == "alphavantage":
        return AlphaVantageSource()
    if name == "csv":
        return CSVSource(kw["path"])
    raise ValueError(f"unknown data source {name!r}")
