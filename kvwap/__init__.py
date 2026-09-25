"""Kalman-filter intraday volume forecasting and VWAP execution.

Implements Chen, Feng & Palomar, "Forecasting Intraday Trading Volume: A
Kalman Filter Approach" (SSRN 3101695): state-space model, closed-form EM
calibration, Lasso-robust filtering, and static/dynamic VWAP replication.
"""
from .model import KalmanParams, KalmanVolumeModel, VolumeState, fit_em, forecast_log_volume, kalman_filter, rts_smoother

__all__ = [
    "KalmanParams",
    "KalmanVolumeModel",
    "VolumeState",
    "fit_em",
    "forecast_log_volume",
    "kalman_filter",
    "rts_smoother",
]
__version__ = "0.1.0"
