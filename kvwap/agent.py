"""VWAP execution agent driven by the Kalman volume forecaster.

One parent order (symbol, side, quantity) is worked over one trading day:

1. Pre-open: calibrate the (robust) Kalman model by EM on the last
   ``train_days`` complete days and compute the static volume profile.
2. At the start of every bin i: pull today's bars, run the filter over the
   bins observed so far (missing/late bins are skipped, not guessed), forecast
   the remaining bins and size the child order
     * dynamic (Eq. 41): remaining_qty * f_i / sum_{j>=i} f_j
     * static  (Eq. 40): follow the pre-open cumulative weights
3. Apply risk caps (participation, notional, kill switch) and send the child
   order, optionally split into equal slices across the bin.
4. After the close: benchmark the fills against the market VWAP and log the
   forecast accuracy to the journal.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .broker import Broker, OrderResult
from .connectors import BarSource
from .data import Session, bars_to_grid, detect_right_labeled, intraday_bins
from .journal import Journal
from .model import KalmanParams, VolumeState, advance, fit_em, forecast_log_volume
from .risk import RiskLimits, check_parent, clamp_child, kill_switch_engaged
from .vwap import mape, slippage_bps, static_weights

TERMINAL_UNFILLED = {"canceled", "expired", "rejected", "done_for_day", "suspended"}


@dataclass
class ParentOrder:
    symbol: str
    side: str  # "buy" | "sell"
    quantity: int

    def __post_init__(self):
        self.side = self.side.lower()
        if self.side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        self.quantity = int(self.quantity)


@dataclass
class AgentConfig:
    order: ParentOrder
    session: Session = field(default_factory=Session)
    mode: str = "dynamic"  # "dynamic" (Eq. 41) or "static" (Eq. 40)
    lam: Optional[float] = 3.0  # robust filter threshold; None = standard Kalman filter
    lam_mode: str = "sigma"
    train_days: int = 40
    min_train_days: int = 10
    em_max_iter: int = 100
    slices_per_bin: int = 1
    bin_delay_seconds: float = 10.0  # wait after the bin opens so the previous bar is published
    risk: RiskLimits = field(default_factory=RiskLimits)

    def __post_init__(self):
        if self.mode not in ("dynamic", "static"):
            raise ValueError("mode must be 'dynamic' or 'static'")
        if self.slices_per_bin < 1:
            raise ValueError("slices_per_bin must be >= 1")


class SystemClock:
    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC")

    def sleep_until(self, ts: pd.Timestamp) -> None:
        while True:
            left = (ts - self.now()).total_seconds()
            if left <= 0:
                return
            _time.sleep(min(left, 30.0))


class SimClock:
    """Deterministic clock for replays: sleeping just jumps forward."""

    def __init__(self, start: pd.Timestamp):
        self.t = pd.Timestamp(start).tz_convert("UTC")

    def now(self) -> pd.Timestamp:
        return self.t

    def sleep_until(self, ts: pd.Timestamp) -> None:
        self.t = max(self.t, pd.Timestamp(ts).tz_convert("UTC"))


@dataclass
class BinRecord:
    bin: int
    forecast_shares: float
    child_target: int
    child_sent: int
    reasons: list


class VWAPExecutionAgent:
    def __init__(self, cfg: AgentConfig, source: BarSource, broker: Broker, clock=None, journal: Optional[Journal] = None):
        self.cfg = cfg
        self.source = source
        self.broker = broker
        self.clock = clock or SystemClock()
        self.journal = journal or Journal()
        self.params: Optional[KalmanParams] = None
        self.state0: Optional[VolumeState] = None
        self.static_log: Optional[np.ndarray] = None
        self.right_labeled = False
        self.ref_price: Optional[float] = None
        self.orders: list[OrderResult] = []
        self.order_bins: list[int] = []
        self.bins: list[BinRecord] = []
        self.dynamic_log = np.full(cfg.session.bins_per_day, np.nan)
        self.day = None

    # ------------------------------------------------------------------ #
    @property
    def I(self) -> int:
        return self.cfg.session.bins_per_day

    def _bin_start(self, i: int) -> pd.Timestamp:
        return self.cfg.session.bin_start_ts(self.day, i).tz_convert("UTC")

    def _bin_end(self, i: int) -> pd.Timestamp:
        return self._bin_start(i) + pd.Timedelta(minutes=self.cfg.session.bin_minutes)

    # ------------------------------------------------------------------ #
    def calibrate(self, day) -> dict:
        """Fit the model on history strictly before ``day`` and plan the day."""
        cfg = self.cfg
        self.day = pd.Timestamp(day).date()
        open_ts = self._bin_start(0)
        lookback = pd.Timedelta(days=int(cfg.train_days * 7 / 5) + 14)
        bars = self.source.get_bars(cfg.order.symbol, open_ts - lookback, open_ts)
        if bars.empty:
            raise RuntimeError(
                f"no historical bars for {cfg.order.symbol} between {open_ts - lookback} and {open_ts} "
                "(stale CSV? pass a --date inside its range)"
            )
        idx = pd.DatetimeIndex(bars.index)
        idx = idx.tz_localize(cfg.session.tz) if idx.tz is None else idx.tz_convert(cfg.session.tz)
        self.right_labeled = detect_right_labeled(idx, cfg.session)
        grid = bars_to_grid(bars, cfg.session, right_labeled=self.right_labeled)
        keep = [k for k, d in enumerate(grid.dates) if d < self.day][-cfg.train_days:]
        if len(keep) < cfg.min_train_days:
            raise RuntimeError(f"only {len(keep)} complete days of history; need >= {cfg.min_train_days}")
        grid = grid.slice(keep[0], keep[-1] + 1)
        fit = fit_em(grid.log_volume, lam=cfg.lam, lam_mode=cfg.lam_mode, max_iter=cfg.em_max_iter)
        self.params = fit.params
        # the day being traded may follow a weekend/holiday: still one day step
        self.state0 = VolumeState(fit.filter.next_mean, fit.filter.next_cov, 0)
        self.static_log, _ = forecast_log_volume(self.state0, self.params, self.I)
        self.ref_price = float(grid.price[-1, -1])
        check_parent(cfg.order.quantity, self.ref_price, cfg.risk)
        w = static_weights(np.exp(self.static_log))
        schedule = [
            {"bin": i, "start": str(self._bin_start(i).tz_convert(cfg.session.tz).time()),
             "forecast_volume": float(np.exp(self.static_log[i])), "weight": float(w[i]),
             "planned_qty": float(w[i] * cfg.order.quantity)}
            for i in range(self.I)
        ]
        info = {
            "symbol": cfg.order.symbol, "day": str(self.day), "train_days": len(keep),
            "first_train_day": str(grid.dates[0]), "last_train_day": str(grid.dates[-1]),
            "em_iterations": fit.n_iter, "em_converged": fit.converged,
            "robust_lam": cfg.lam, "params": self.params.to_dict(), "schedule": schedule,
            "forecast_day_volume": float(np.exp(self.static_log).sum()),
        }
        self.journal.log("calibrated", ts=self.clock.now(), **{k: v for k, v in info.items()})
        return info

    # ------------------------------------------------------------------ #
    def _observed_today(self):
        now = self.clock.now()
        bars = self.source.get_bars(self.cfg.order.symbol, self._bin_start(0), now)
        vol, px, vw = intraday_bins(bars, self.cfg.session, self.day, self.right_labeled)
        for j in range(self.I):  # drop bins still in progress
            if self._bin_end(j) > now:
                vol[j] = px[j] = vw[j] = np.nan
        return vol, px, vw

    def _refresh_orders(self) -> None:
        self.orders = [self.broker.refresh(o) for o in self.orders]

    def filled_qty(self) -> int:
        return sum(o.filled_qty for o in self.orders)

    def committed_qty(self) -> int:
        """Shares filled or still working (used for planning the next slice)."""
        return sum(o.filled_qty if o.status in TERMINAL_UNFILLED else o.qty for o in self.orders)

    def plan_bin(self, i: int, vol_today: np.ndarray) -> BinRecord:
        """Size the child order for bin ``i`` given today's observed volumes."""
        cfg = self.cfg
        y = np.log(np.where(vol_today[:i] > 0, vol_today[:i], np.nan))
        state = advance(self.state0, self.params, y, lam=cfg.lam, lam_mode=cfg.lam_mode) if i else self.state0
        m, _ = forecast_log_volume(state, self.params, self.I - i)
        f = np.exp(m)
        self.dynamic_log[i] = m[0]
        total = cfg.order.quantity
        committed = self.committed_qty()
        remaining = total - committed
        last = i == self.I - 1
        if remaining <= 0:
            target = 0
        elif last:
            target = remaining
        elif cfg.mode == "dynamic":
            target = int(round(remaining * f[0] / f.sum()))
        else:
            cum = np.cumsum(static_weights(np.exp(self.static_log)))
            target = int(round(total * cum[i])) - committed
        target = max(min(target, remaining), 0)
        dec = clamp_child(target, f[0], self.ref_price, cfg.risk, last_bin=last)
        return BinRecord(i, float(f[0]), target, dec.qty, dec.reasons)

    def _send(self, i: int, qty: int) -> None:
        """Send ``qty`` for bin ``i`` as ``slices_per_bin`` equal slices across the bin."""
        n = min(self.cfg.slices_per_bin, max(qty, 1))
        sizes = [qty // n + (1 if k < qty % n else 0) for k in range(n)]
        step = pd.Timedelta(minutes=self.cfg.session.bin_minutes) / n
        for k, q in enumerate(sizes):
            if k:
                self.clock.sleep_until(self._bin_start(i) + k * step)
            if q <= 0:
                continue
            if kill_switch_engaged(self.cfg.risk):
                self.journal.log("kill_switch", ts=self.clock.now(), bin=i, unsent=sum(sizes[k:]))
                return
            o = self.broker.submit_market_order(self.cfg.order.symbol, self.cfg.order.side, q)
            self.orders.append(o)
            self.order_bins.append(i)
            self.journal.log("order", ts=self.clock.now(), bin=i, slice=k, qty=q, order_id=o.order_id,
                             status=o.status, filled_qty=o.filled_qty, avg_price=o.avg_price, message=o.message)

    # ------------------------------------------------------------------ #
    def run_day(self, day=None) -> dict:
        """Calibrate (if needed) and work the parent order through the session."""
        if day is None:
            day = self.clock.now().tz_convert(self.cfg.session.tz).date()
        if self.params is None or self.day != pd.Timestamp(day).date():
            self.calibrate(day)
        if self.clock.now() >= self._bin_end(self.I - 1):
            raise RuntimeError("the session for this day is already over")
        delay = pd.Timedelta(seconds=self.cfg.bin_delay_seconds)
        for i in range(self.I):
            if self.clock.now() >= self._bin_end(i):
                self.journal.log("bin_skipped", ts=self.clock.now(), bin=i, reason="started after bin end")
                continue
            self.clock.sleep_until(self._bin_start(i) + delay)
            if kill_switch_engaged(self.cfg.risk):
                self.journal.log("kill_switch", ts=self.clock.now(), bin=i)
                break
            self._refresh_orders()
            vol, _, _ = self._observed_today()
            rec = self.plan_bin(i, vol)
            self.bins.append(rec)
            self.journal.log("bin", ts=self.clock.now(), bin=i, forecast_volume=round(rec.forecast_shares),
                             target=rec.child_target, send=rec.child_sent, capped_by=rec.reasons,
                             committed=self.committed_qty(), observed_bins=int(np.sum(~np.isnan(vol[:i]))))
            if rec.child_sent > 0:
                self._send(i, rec.child_sent)
        self.clock.sleep_until(self._bin_end(self.I - 1) + delay)
        return self.report()

    def report(self) -> dict:
        self._refresh_orders()
        vol, px, vw = self._observed_today()
        cfg = self.cfg
        filled = self.filled_qty()
        fills = [(o.filled_qty, o.avg_price) for o in self.orders if o.filled_qty and o.avg_price]
        avg_fill = sum(q * p for q, p in fills) / sum(q for q, _ in fills) if fills else None
        ok = ~np.isnan(vol)
        bin_px = np.where(np.isnan(vw), px, vw)
        mkt_vwap = float(np.nansum(bin_px[ok] * vol[ok]) / np.nansum(vol[ok])) if ok.any() else None
        out = {
            "symbol": cfg.order.symbol, "side": cfg.order.side, "day": str(self.day), "mode": cfg.mode,
            "quantity": cfg.order.quantity, "filled": filled,
            "completion_pct": 100.0 * filled / cfg.order.quantity,
            "orders": len(self.orders), "avg_fill_price": avg_fill, "market_vwap": mkt_vwap,
            "slippage_vs_vwap_bps": slippage_bps(avg_fill, mkt_vwap, cfg.order.side) if avg_fill and mkt_vwap else None,
        }
        if ok.all():
            out["static_forecast_mape"] = mape(vol, np.exp(self.static_log))
            dyn_ok = ~np.isnan(self.dynamic_log)
            if dyn_ok.any():
                out["dynamic_forecast_mape"] = mape(vol[dyn_ok], np.exp(self.dynamic_log[dyn_ok]))
            # replication error of the realised schedule, paper-style (last price per bin)
            per_bin = np.zeros(self.I)
            for o, b in zip(self.orders, self.order_bins):
                per_bin[b] += o.filled_qty
            if per_bin.sum() > 0:
                true_vwap = float(np.sum(px * vol) / np.sum(vol))
                rep = float(np.sum(px * per_bin) / per_bin.sum())
                out["schedule_tracking_error_bps"] = abs(rep - true_vwap) / true_vwap * 1e4
        self.journal.log("summary", ts=self.clock.now(), **out)
        return out


def simulate_day(cfg: AgentConfig, bars: pd.DataFrame, day, cost_bps: float = 0.0,
                 journal: Optional[Journal] = None, feed_delay_minutes: int = 0) -> dict:
    """Replay ``day`` from historical bars: the agent sees only finished bars
    (optionally delayed) and fills at each bin's VWAP (or last price) proxy."""
    from .broker import SimulatedBroker
    from .connectors import ReplaySource

    start = cfg.session.bin_start_ts(day, 0).tz_convert("UTC") - pd.Timedelta(minutes=5)
    clock = SimClock(start)
    src = ReplaySource(bars, clock.now, bar_minutes=_native_minutes(bars, cfg.session), delay_minutes=feed_delay_minutes)
    broker = SimulatedBroker(lambda: src.bar_price(clock.now()), cost_bps=cost_bps)
    agent = VWAPExecutionAgent(cfg, src, broker, clock, journal or Journal(echo=False))
    return agent.run_day(day)


def _native_minutes(bars: pd.DataFrame, session: Session) -> int:
    d = pd.Series(pd.DatetimeIndex(bars.index)).diff().dropna()
    return int(d.min().total_seconds() // 60) if not d.empty else session.bin_minutes
