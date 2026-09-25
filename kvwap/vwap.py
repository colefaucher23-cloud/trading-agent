"""VWAP replication and evaluation metrics (paper Sec. 4.3, Eqs. 37-42)."""
from __future__ import annotations

import numpy as np


def static_weights(volume_forecast: np.ndarray) -> np.ndarray:
    """Eq. 40: slice the order in proportion to the full-day volume forecast."""
    v = np.asarray(volume_forecast, float)
    return v / v.sum()


def dynamic_weight(remaining_forecast: np.ndarray, executed_fraction: float) -> float:
    """Eq. 41 for one bin.

    ``remaining_forecast`` holds the latest volume forecasts for the current bin
    and every bin after it; the current bin gets its share of the fraction of
    the order still left to trade. On the last bin this returns the remainder.
    """
    f = np.asarray(remaining_forecast, float)
    left = max(1.0 - executed_fraction, 0.0)
    if f.size == 1:
        return left
    return float(f[0] / f.sum() * left)


def vwap(price: np.ndarray, volume: np.ndarray) -> float:
    """Eq. 39, with the last price in each bin as a proxy for its VWAP."""
    price = np.asarray(price, float)
    volume = np.asarray(volume, float)
    return float(np.sum(price * volume) / np.sum(volume))


def replicated_vwap(price: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(np.asarray(price, float) * np.asarray(weights, float)))


def tracking_error_bps(true_vwap: float, rep_vwap: float) -> float:
    """One day of Eq. 42, in basis points."""
    return abs(true_vwap - rep_vwap) / true_vwap * 1e4


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Eq. 37, evaluated on volume (not log-volume)."""
    a = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    return float(np.mean(np.abs(a - p) / a))


def improvement(err_benchmark: float, err_model: float) -> float:
    """Eq. 38, relative improvement in percent."""
    return 100.0 * (err_benchmark - err_model) / err_benchmark


def slippage_bps(avg_fill: float, benchmark: float, side: str) -> float:
    """Execution cost against a benchmark price; positive = worse than benchmark."""
    sign = 1.0 if side == "buy" else -1.0
    return sign * (avg_fill - benchmark) / benchmark * 1e4
