"""Turn raw intraday bars into the (days x bins) grid the model works on."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as _dtime
from typing import Optional, Union

import numpy as np
import pandas as pd

from .model import KalmanParams


@dataclass
class Session:
    """Regular trading hours of an exchange, split into fixed-width bins."""

    tz: str = "America/New_York"
    open: str = "09:30"
    close: str = "16:00"
    bin_minutes: int = 15

    @property
    def bins_per_day(self) -> int:
        o, c = _to_minutes(self.open), _to_minutes(self.close)
        if (c - o) % self.bin_minutes:
            raise ValueError("session length must be a multiple of bin_minutes")
        return (c - o) // self.bin_minutes

    def bin_starts(self) -> list:
        o = _to_minutes(self.open)
        return [_dtime((o + k * self.bin_minutes) // 60, (o + k * self.bin_minutes) % 60) for k in range(self.bins_per_day)]

    def bin_start_ts(self, day, i: int) -> pd.Timestamp:
        """Timezone-aware start timestamp of bin ``i`` on ``day``."""
        o = _to_minutes(self.open) + i * self.bin_minutes
        return pd.Timestamp(pd.Timestamp(day).date()).tz_localize(self.tz) + pd.Timedelta(minutes=o)


def _to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


@dataclass
class DailyGrid:
    """Intraday bars on a regular grid: row = trading day, column = bin."""

    dates: list
    volume: np.ndarray  # (T, I) shares (or turnover if normalised)
    price: np.ndarray  # (T, I) last traded price in the bin
    session: Session

    @property
    def log_volume(self) -> np.ndarray:
        return np.log(self.volume)

    @property
    def n_days(self) -> int:
        return len(self.dates)

    def slice(self, start: int, stop: int) -> "DailyGrid":
        return DailyGrid(self.dates[start:stop], self.volume[start:stop], self.price[start:stop], self.session)


def detect_right_labeled(index: pd.DatetimeIndex, session: Session) -> bool:
    """True if bar timestamps mark the bar *end* (e.g. 16:00 present, 09:30 absent)."""
    tod = index.strftime("%H:%M")
    has_open, has_close = (tod == session.open).any(), (tod == session.close).any()
    return bool(has_close and not has_open)


def _binned(bars: pd.DataFrame, session: Session, right_labeled: bool) -> pd.DataFrame:
    """Aggregate bars into session bins: columns close, volume, vwap?, date, bin."""
    df = bars.copy()
    df.columns = [c.lower() for c in df.columns]
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize(session.tz) if idx.tz is None else idx.tz_convert(session.tz)
    if right_labeled:
        # shift to bar-start labels using the native bar width
        width = pd.Series(df.index).diff().dropna().min()
        if pd.isna(width):
            width = pd.Timedelta(minutes=session.bin_minutes)
        df.index = df.index - width
    spec = {"close": "last", "volume": "sum"}
    has_vwap = "vwap" in df.columns
    if has_vwap:
        df["pv"] = df["vwap"].astype(float) * df["volume"].astype(float)
        spec["pv"] = "sum"
    agg = df.resample(f"{session.bin_minutes}min", label="left", closed="left").agg(spec)
    agg = agg.dropna(subset=["close"])
    if has_vwap:
        agg["vwap"] = agg["pv"] / agg["volume"].where(agg["volume"] > 0)
        agg = agg.drop(columns="pv")
    starts = [t.strftime("%H:%M") for t in session.bin_starts()]
    tod = agg.index.strftime("%H:%M")
    agg = agg[np.isin(tod, starts)].copy()
    agg["date"] = agg.index.date
    agg["bin"] = [starts.index(t) for t in agg.index.strftime("%H:%M")]
    return agg


def bars_to_grid(
    bars: pd.DataFrame,
    session: Session = Session(),
    right_labeled: Optional[bool] = None,
    shares_outstanding: Union[None, float, pd.Series] = None,
    drop_incomplete: bool = True,
) -> DailyGrid:
    """Resample bars to ``session.bin_minutes`` and build a DailyGrid.

    ``bars`` needs a DatetimeIndex (naive = already in exchange time, aware =
    converted) and ``close``/``volume`` columns. Days missing any regular
    session bin (half days, halts, gaps) are dropped, as in the paper. Zero
    volume bins are floored at half the smallest positive volume so log() is
    defined (the paper assumes non-zero volume; the robust filter absorbs the
    resulting outlier).

    ``shares_outstanding`` optionally normalises volume into turnover (Eq. 1).
    """
    if bars.empty:
        raise ValueError("no bars supplied")
    if right_labeled is None:
        idx = pd.DatetimeIndex(bars.index)
        idx = idx.tz_localize(session.tz) if idx.tz is None else idx.tz_convert(session.tz)
        right_labeled = detect_right_labeled(idx, session)
    agg = _binned(bars, session, right_labeled)
    I = session.bins_per_day
    vol = agg.pivot(index="date", columns="bin", values="volume").reindex(columns=range(I))
    px = agg.pivot(index="date", columns="bin", values="close").reindex(columns=range(I))
    if drop_incomplete:
        keep = vol.notna().all(axis=1) & px.notna().all(axis=1)
        vol, px = vol[keep], px[keep]
    if vol.empty:
        raise ValueError("no complete trading days in the supplied bars")

    v = vol.to_numpy(float)
    pos = v[v > 0]
    if pos.size == 0:
        raise ValueError("all volumes are zero")
    v = np.where(v > 0, v, 0.5 * pos.min())
    if shares_outstanding is not None:
        if isinstance(shares_outstanding, pd.Series):
            so = shares_outstanding.reindex(vol.index).ffill().bfill().to_numpy(float)
            v = v / so[:, None]
        else:
            v = v / float(shares_outstanding)
    return DailyGrid(dates=list(vol.index), volume=v, price=px.to_numpy(float), session=session)


def intraday_bins(bars: pd.DataFrame, session: Session, day, right_labeled: bool = False):
    """Per-bin (volume, close, vwap) arrays for one day; NaN where no bar yet."""
    I = session.bins_per_day
    vol, px, vw = np.full(I, np.nan), np.full(I, np.nan), np.full(I, np.nan)
    if bars is None or bars.empty:
        return vol, px, vw
    agg = _binned(bars, session, right_labeled)
    agg = agg[agg["date"] == pd.Timestamp(day).date()]
    for _, row in agg.iterrows():
        b = int(row["bin"])
        vol[b], px[b] = row["volume"], row["close"]
        if "vwap" in agg.columns and pd.notna(row["vwap"]):
            vw[b] = row["vwap"]
    return vol, px, vw


def load_csv(path: str) -> pd.DataFrame:
    """Read bars saved by ``kvwap fetch`` (timestamp index, close/volume columns)."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


# --------------------------------------------------------------------------- #
# Synthetic data (used by tests and the offline demo)
# --------------------------------------------------------------------------- #
def u_shape_phi(I: int, level: float = 11.0, depth: float = 0.9) -> np.ndarray:
    """A typical U-shaped intraday seasonal profile in log-volume."""
    x = np.linspace(-1.0, 1.0, I)
    phi = level + depth * (x**2 - 1.0 / 3.0)
    phi[-1] += 0.5  # closing-auction bump
    return phi


def default_true_params(I: int = 26) -> KalmanParams:
    return KalmanParams(
        pi1=np.array([0.0, 0.0]),
        sigma1=np.diag([0.05, 0.05]),
        a_eta=0.95,
        a_mu=0.7,
        sigma_eta2=0.04,
        sigma_mu2=0.05,
        r=0.04,
        phi=u_shape_phi(I),
    )


def simulate_log_volume(
    params: KalmanParams,
    n_days: int,
    seed: int = 0,
    outlier_frac: float = 0.0,
    outlier_scale: float = 0.0,
):
    """Draw (clean, contaminated) log-volume grids from the state-space model.

    Outliers follow Sec. 3.3: a random ``outlier_frac`` of bins receive a
    sparse shock z ~ +/- U(0.5, 1) * ``outlier_scale`` (in log units).
    """
    rng = np.random.default_rng(seed)
    I = params.bins_per_day
    N = n_days * I
    x = rng.multivariate_normal(params.pi1, params.sigma1)
    y = np.empty(N)
    for t in range(N):
        y[t] = x[0] + x[1] + params.phi[t % I] + rng.normal(0.0, np.sqrt(params.r))
        if (t + 1) % I == 0:
            x[0] = params.a_eta * x[0] + rng.normal(0.0, np.sqrt(params.sigma_eta2))
        x[1] = params.a_mu * x[1] + rng.normal(0.0, np.sqrt(params.sigma_mu2))
    clean = y.reshape(n_days, I)
    dirty = clean.copy()
    if outlier_frac > 0 and outlier_scale > 0:
        mask = rng.random(N) < outlier_frac
        shock = rng.choice([-1.0, 1.0], N) * rng.uniform(0.5, 1.0, N) * outlier_scale
        dirty = (y + mask * shock).reshape(n_days, I)
    return clean, dirty


def synthetic_bars(
    n_days: int = 120,
    session: Session = Session(),
    seed: int = 0,
    params: Optional[KalmanParams] = None,
    start: str = "2026-01-05",
    daily_vol: float = 0.012,
) -> pd.DataFrame:
    """Synthetic 15-min bars (close, volume) on business days, for demos/tests."""
    I = session.bins_per_day
    params = params or default_true_params(I)
    clean, _ = simulate_log_volume(params, n_days, seed=seed)
    rng = np.random.default_rng(seed + 1)
    days = pd.bdate_range(start, periods=n_days)
    step_sd = daily_vol / np.sqrt(I)
    logp = np.log(100.0) + np.cumsum(rng.normal(0.0, step_sd, n_days * I))
    rows = []
    for d, day in enumerate(days):
        for i in range(I):
            rows.append((session.bin_start_ts(day, i), float(np.exp(logp[d * I + i])), float(np.round(np.exp(clean[d, i])))))
    df = pd.DataFrame(rows, columns=["timestamp", "close", "volume"]).set_index("timestamp")
    return df
