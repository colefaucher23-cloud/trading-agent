"""Rolling-window out-of-sample evaluation and cross-validation (paper Sec. 4).

For every out-of-sample day the models are re-calibrated on the previous
``train_days`` days (warm-started from the previous day's parameters), then

* static prediction  -- all I bins of the day from information up to the prior close;
* dynamic prediction -- one-bin-ahead, correcting the filter with each new bin.

Both feed the static (Eq. 40) and dynamic (Eq. 41) VWAP replication strategies.
The paper re-estimates on the previous N *bins* for every bin; re-calibrating
once per day is the standard practical approximation (parameters move slowly).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .data import DailyGrid
from .model import KalmanParams, advance, fit_em, forecast_log_volume, VolumeState
from .vwap import dynamic_weight, mape, replicated_vwap, static_weights, tracking_error_bps, vwap


@dataclass
class BacktestConfig:
    train_days: int = 40
    lam: Optional[float] = 3.0  # robust Kalman filter threshold (sigma units by default)
    lam_mode: str = "sigma"
    rm_days: Optional[int] = None  # rolling-means window; defaults to train_days
    em_max_iter: int = 100
    em_warm_iter: int = 30
    models: Sequence[str] = ("rm", "kf", "rkf")


@dataclass
class DayForecast:
    static_log: np.ndarray
    dynamic_log: np.ndarray
    static_w: np.ndarray
    dynamic_w: np.ndarray


def day_forecasts(
    params: KalmanParams,
    state0: VolumeState,
    observed_log: np.ndarray,
    lam: Optional[float],
    lam_mode: str = "sigma",
) -> DayForecast:
    """Static + dynamic forecasts and VWAP weights for one day."""
    I = params.bins_per_day
    static_log, _ = forecast_log_volume(state0, params, I)
    dyn = np.empty(I)
    dyn_w = np.empty(I)
    executed = 0.0
    state = state0
    for i in range(I):
        m, _ = forecast_log_volume(state, params, I - i)
        dyn[i] = m[0]
        dyn_w[i] = dynamic_weight(np.exp(m), executed)
        executed += dyn_w[i]
        state = advance(state, params, observed_log[i], lam=lam, lam_mode=lam_mode)
    return DayForecast(static_log, dyn, static_weights(np.exp(static_log)), dyn_w)


@dataclass
class BacktestResult:
    config: BacktestConfig
    dates: list
    # per model/mode -> per-day arrays
    mape: dict = field(default_factory=dict)
    vwap_te_bps: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)  # last calibrated params per model

    def summary(self) -> dict:
        out = {"days": len(self.dates), "mape": {}, "vwap_te_bps": {}}
        for k, v in self.mape.items():
            out["mape"][k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in self.vwap_te_bps.items():
            out["vwap_te_bps"][k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
        return out

    def table(self) -> str:
        s = self.summary()
        rows = [f"{'model':<14}{'MAPE':>10}{'VWAP TE (bps)':>16}"]
        for k in sorted(set(s["mape"]) | set(s["vwap_te_bps"])):
            m = s["mape"].get(k, {}).get("mean", float("nan"))
            v = s["vwap_te_bps"].get(k, {}).get("mean", float("nan"))
            rows.append(f"{k:<14}{m:>10.4f}{v:>16.3f}")
        return "\n".join(rows)


def rolling_backtest(
    grid: DailyGrid,
    cfg: BacktestConfig = BacktestConfig(),
    start: Optional[int] = None,
    stop: Optional[int] = None,
    observed_log: Optional[np.ndarray] = None,
    progress: bool = False,
) -> BacktestResult:
    """Evaluate RM / KF / robust KF on days ``start`` .. ``stop - 1`` of ``grid``.

    ``observed_log`` (default: log of grid volume) is what the models see for
    calibration and filtering; scoring always uses ``grid.volume``. Passing a
    contaminated copy reproduces the outlier experiment of Sec. 3.3.
    """
    obs = grid.log_volume if observed_log is None else np.asarray(observed_log, float)
    T, I = grid.volume.shape
    start = cfg.train_days if start is None else start
    stop = T if stop is None else stop
    if start < cfg.train_days:
        raise ValueError("start must leave train_days of history")
    if start >= stop:
        raise ValueError("empty evaluation window")
    rm_days = cfg.rm_days or cfg.train_days

    res = BacktestResult(config=cfg, dates=grid.dates[start:stop])
    kf_models = {"kf": None, "rkf": cfg.lam}
    kf_models = {k: v for k, v in kf_models.items() if k in cfg.models}
    warm: dict = {}

    def push(d, key, val):
        d.setdefault(key, []).append(val)

    for d in range(start, stop):
        actual = grid.volume[d]
        price = grid.price[d]
        true_vwap = vwap(price, actual)
        if "rm" in cfg.models:
            rm = np.exp(obs[d - rm_days:d]).mean(axis=0)
            push(res.mape, "rm_static", mape(actual, rm))
            push(res.vwap_te_bps, "rm_static", tracking_error_bps(true_vwap, replicated_vwap(price, static_weights(rm))))
        hist = obs[d - cfg.train_days:d]
        for name, lam in kf_models.items():
            init = warm.get(name)
            fit = fit_em(
                hist,
                lam=lam,
                lam_mode=cfg.lam_mode,
                init=init,
                max_iter=cfg.em_warm_iter if init is not None else cfg.em_max_iter,
            )
            warm[name] = fit.params
            st = VolumeState(fit.filter.next_mean, fit.filter.next_cov, fit.filter.next_bin)
            fc = day_forecasts(fit.params, st, obs[d], lam, cfg.lam_mode)
            push(res.mape, f"{name}_static", mape(actual, np.exp(fc.static_log)))
            push(res.mape, f"{name}_dynamic", mape(actual, np.exp(fc.dynamic_log)))
            push(res.vwap_te_bps, f"{name}_static", tracking_error_bps(true_vwap, replicated_vwap(price, fc.static_w)))
            push(res.vwap_te_bps, f"{name}_dynamic", tracking_error_bps(true_vwap, replicated_vwap(price, fc.dynamic_w)))
        if progress:
            print(f"  {grid.dates[d]}  done ({d - start + 1}/{stop - start})", flush=True)
    res.params = {k: v.to_dict() for k, v in warm.items()}
    for k in list(res.mape):
        res.mape[k] = np.array(res.mape[k])
    for k in list(res.vwap_te_bps):
        res.vwap_te_bps[k] = np.array(res.vwap_te_bps[k])
    return res


@dataclass
class CVResult:
    best_train_days: int
    best_lam: Optional[float]
    scores: list  # (train_days, lam, dynamic MAPE)


def cross_validate(
    grid: DailyGrid,
    cv_days: int,
    test_start: Optional[int] = None,
    train_days_grid: Sequence[int] = (20, 40, 60),
    lam_grid: Sequence[Optional[float]] = (None, 2.0, 3.0, 4.0),
    lam_mode: str = "sigma",
    em_warm_iter: int = 20,
    observed_log: Optional[np.ndarray] = None,
) -> CVResult:
    """Pick (train_days, lam) by dynamic-prediction MAPE on a CV window.

    The CV window is the ``cv_days`` days right before ``test_start``
    (default: the last ``cv_days`` days of the grid). ``lam=None`` is the
    standard Kalman filter, so CV also decides whether robustness helps.
    """
    T = grid.n_days
    test_start = T if test_start is None else test_start
    cv_start = test_start - cv_days
    scores = []
    for n, lam in itertools.product(train_days_grid, lam_grid):
        if cv_start < n:
            continue
        cfg = BacktestConfig(
            train_days=n, lam=lam, lam_mode=lam_mode, em_warm_iter=em_warm_iter,
            models=("rkf",) if lam is not None else ("kf",),
        )
        r = rolling_backtest(grid, cfg, start=cv_start, stop=test_start, observed_log=observed_log)
        key = "rkf_dynamic" if lam is not None else "kf_dynamic"
        scores.append((n, lam, float(np.mean(r.mape[key]))))
    if not scores:
        raise ValueError("not enough history for any train_days in the grid")
    best = min(scores, key=lambda s: s[2])
    return CVResult(best_train_days=best[0], best_lam=best[1], scores=scores)
