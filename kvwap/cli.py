"""Command line interface: ``python -m kvwap <command>`` (or ``kvwap`` once installed)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def _session(args):
    from .data import Session

    return Session(tz=args.tz, open=args.open, close=args.close, bin_minutes=args.bin_minutes)


def _add_session_args(p):
    p.add_argument("--tz", default="America/New_York")
    p.add_argument("--open", default="09:30")
    p.add_argument("--close", default="16:00")
    p.add_argument("--bin-minutes", type=int, default=15)


def _lam(v: str):
    return None if v.lower() in ("none", "off", "0") else float(v)


# --------------------------------------------------------------------------- #
def cmd_fetch(args):
    from .connectors import make_source

    src = make_source(args.source, feed=args.feed)
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=args.days)
    bars = src.get_bars(args.symbol, start, end)
    if bars.empty:
        sys.exit("no bars returned")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bars.to_csv(args.out)
    print(f"saved {len(bars)} bars {bars.index[0]} .. {bars.index[-1]} -> {args.out}")


def _load_grid(args):
    from .data import bars_to_grid, load_csv, synthetic_bars

    session = _session(args)
    bars = synthetic_bars(n_days=args.synthetic_days, session=session, seed=args.seed) if args.csv is None else load_csv(args.csv)
    return bars, bars_to_grid(bars, session)


def cmd_backtest(args):
    from .evaluate import BacktestConfig, cross_validate, rolling_backtest

    _, grid = _load_grid(args)
    T = grid.n_days
    print(f"{T} complete days of {grid.session.bins_per_day} bins ({grid.dates[0]} .. {grid.dates[-1]})")
    test_days = args.test_days or max(T - args.train_days - args.cv_days, 1)
    test_start = T - test_days
    train_days, lam = args.train_days, args.lam
    if args.cv_days:
        print(f"cross-validating train_days x lam on {args.cv_days} days before the test window ...")
        tgrid = [n for n in (10, 20, 30, 40, 60) if n <= test_start - args.cv_days] or [test_start - args.cv_days]
        cv = cross_validate(grid, args.cv_days, test_start, train_days_grid=tgrid, lam_grid=(None, 2.0, 3.0, 4.0))
        for n, l, s in cv.scores:
            print(f"  train_days={n:<3} lam={str(l):<5} dynamic MAPE={s:.4f}")
        train_days, lam = cv.best_train_days, cv.best_lam
        print(f"selected train_days={train_days} lam={lam}")
    if test_start < train_days:
        sys.exit(f"not enough history: need > {train_days} days before the test window")
    cfg = BacktestConfig(train_days=train_days, lam=lam if lam is not None else 3.0,
                         models=("rm", "kf", "rkf") if lam is not None else ("rm", "kf"))
    print(f"out-of-sample: {test_days} days, re-calibrated daily on the previous {train_days} days")
    res = rolling_backtest(grid, cfg, start=test_start, stop=T, progress=args.verbose)
    print(res.table())
    s = res.summary()
    m = s["mape"]
    best = "rkf_dynamic" if "rkf_dynamic" in m else "kf_dynamic"
    print(f"\n{best} vs RM: MAPE improvement {100 * (1 - m[best]['mean'] / m['rm_static']['mean']):.1f}%, "
          f"VWAP TE improvement {100 * (1 - s['vwap_te_bps'][best]['mean'] / s['vwap_te_bps']['rm_static']['mean']):.1f}%")
    if args.json:
        s["train_days"], s["lam"], s["params"] = train_days, lam, res.params
        with open(args.json, "w") as fh:
            json.dump(s, fh, indent=2)
        print(f"wrote {args.json}")


def _build_run(cfg_path, dry_run_override=False):
    from .broker import AlpacaBroker, DryRunBroker
    from .config import load_config
    from .connectors import make_source

    rc = load_config(cfg_path)
    source = make_source(rc.data_source, **rc.data_options)
    broker_name = "dry-run" if dry_run_override else rc.broker
    if broker_name == "dry-run":
        broker = DryRunBroker()
    elif broker_name == "alpaca-paper":
        broker = AlpacaBroker(live=False)
    elif broker_name == "alpaca-live":
        broker = AlpacaBroker(live=True)
    else:
        raise ValueError(f"unknown broker {broker_name!r}")
    return rc, source, broker


def cmd_forecast(args):
    from .agent import VWAPExecutionAgent
    from .broker import DryRunBroker
    from .journal import Journal

    rc, source, _ = _build_run(args.config, dry_run_override=True)
    day = pd.Timestamp(args.date).date() if args.date else pd.Timestamp.now(tz=rc.agent.session.tz).date()
    agent = VWAPExecutionAgent(rc.agent, source, DryRunBroker(), journal=Journal(echo=False))
    info = agent.calibrate(day)
    print(f"{info['symbol']} {info['day']}: trained on {info['train_days']} days "
          f"({info['first_train_day']} .. {info['last_train_day']}), EM iterations {info['em_iterations']}")
    p = info["params"]
    print(f"a_eta={p['a_eta']:.3f} a_mu={p['a_mu']:.3f} sigma_eta2={p['sigma_eta2']:.4f} "
          f"sigma_mu2={p['sigma_mu2']:.4f} r={p['r']:.4f}")
    print(f"forecast day volume: {info['forecast_day_volume']:,.0f}\n")
    print(f"{'bin':>3} {'start':>8} {'fcst volume':>14} {'weight':>8} {'planned qty':>12}")
    for row in info["schedule"]:
        print(f"{row['bin']:>3} {row['start'][:5]:>8} {row['forecast_volume']:>14,.0f} {row['weight']:>8.4f} {row['planned_qty']:>12,.0f}")


def cmd_run(args):
    from .agent import SystemClock, VWAPExecutionAgent
    from .journal import Journal

    rc, source, broker = _build_run(args.config, dry_run_override=args.dry_run)
    journal = Journal(rc.journal_path)
    day = pd.Timestamp.now(tz=rc.agent.session.tz).date()
    if hasattr(broker, "calendar"):
        cal = broker.calendar(day)
        if cal is None:
            sys.exit(f"{day} is not a trading day")
        if cal.get("close") and cal["close"] != rc.agent.session.close:
            sys.exit(f"{day} closes early ({cal['close']}); half days are excluded by the strategy")
    elif pd.Timestamp(day).weekday() >= 5:
        sys.exit(f"{day} is a weekend")
    agent = VWAPExecutionAgent(rc.agent, source, broker, SystemClock(), journal)
    summary = agent.run_day(day)
    print(json.dumps(summary, indent=2, default=float))


def cmd_simulate(args):
    from .agent import simulate_day
    from .config import load_config
    from .data import load_csv, synthetic_bars
    from .journal import Journal

    rc = load_config(args.config)
    session = rc.agent.session
    bars = synthetic_bars(n_days=args.synthetic_days, session=session, seed=args.seed) if args.csv is None else load_csv(args.csv)
    days = sorted(set(pd.DatetimeIndex(bars.index).tz_convert(session.tz).date))
    day = pd.Timestamp(args.date).date() if args.date else days[-1]
    journal = Journal(args.journal, echo=args.verbose)
    s = simulate_day(rc.agent, bars, day, cost_bps=args.cost_bps, journal=journal, feed_delay_minutes=args.feed_delay)
    print(json.dumps(s, indent=2, default=float))


def cmd_demo(args):
    """Offline end-to-end run on synthetic data from the paper's own model."""
    from .agent import AgentConfig, ParentOrder, simulate_day
    from .data import DailyGrid, Session, bars_to_grid, default_true_params, simulate_log_volume, synthetic_bars
    from .evaluate import BacktestConfig, rolling_backtest
    from .journal import Journal
    from .risk import RiskLimits

    session = Session()
    bars = synthetic_bars(n_days=90, session=session, seed=args.seed)
    grid = bars_to_grid(bars, session)
    print("1) Out-of-sample backtest on clean synthetic data (40-day window, 30 test days)")
    res = rolling_backtest(grid, BacktestConfig(train_days=40, lam=3.0), start=60)
    print(res.table())

    print("\n2) Same, with 10% of bins contaminated by outliers (models see dirty data, scored on clean)")
    clean, dirty = simulate_log_volume(default_true_params(session.bins_per_day), 90, seed=args.seed,
                                       outlier_frac=0.10, outlier_scale=3.0)
    g2 = DailyGrid(grid.dates, np.exp(clean), grid.price, session)
    res2 = rolling_backtest(g2, BacktestConfig(train_days=40, lam=2.0), start=60, observed_log=dirty)
    print(res2.table())

    print("\n3) Agent replay of the last day: buy 150,000 shares, dynamic VWAP, 3 slices per bin, 1 bps cost")
    cfg = AgentConfig(order=ParentOrder("SYNTH", "buy", 150_000), session=session, slices_per_bin=3,
                      risk=RiskLimits(max_participation=0.25))
    s = simulate_day(cfg, bars, grid.dates[-1], cost_bps=1.0, journal=Journal(echo=False))
    print(json.dumps(s, indent=2, default=float))


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(prog="kvwap", description="Kalman-filter intraday volume forecasting + VWAP execution agent")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch", help="download 15-min bars to CSV")
    p.add_argument("--source", default="yahoo", choices=["yahoo", "alpaca", "alphavantage"])
    p.add_argument("--feed", default="iex", help="alpaca feed: iex (free, real-time) or sip (free, 15-min delayed)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--days", type=int, default=59)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("backtest", help="rolling out-of-sample MAPE / VWAP tracking error vs rolling means")
    p.add_argument("--csv", help="bars CSV (omit to use synthetic data)")
    p.add_argument("--synthetic-days", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-days", type=int, default=40)
    p.add_argument("--lam", type=_lam, default=3.0, help="robust threshold in std devs; 'none' = standard KF")
    p.add_argument("--cv-days", type=int, default=0, help="cross-validate train_days and lam on this many days")
    p.add_argument("--test-days", type=int, default=0)
    p.add_argument("--json", help="write summary JSON here")
    p.add_argument("--verbose", action="store_true")
    _add_session_args(p)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("forecast", help="calibrate and print today's volume forecast + VWAP schedule")
    p.add_argument("--config", required=True)
    p.add_argument("--date")
    p.set_defaults(func=cmd_forecast)

    p = sub.add_parser("run", help="work the configured parent order through today's session")
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true", help="log orders without sending them")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("simulate", help="replay the agent on a historical day")
    p.add_argument("--config", required=True)
    p.add_argument("--csv", help="bars CSV (omit to use synthetic data)")
    p.add_argument("--synthetic-days", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--date")
    p.add_argument("--cost-bps", type=float, default=1.0)
    p.add_argument("--feed-delay", type=int, default=0, help="minutes of data delay (e.g. 15 for free SIP)")
    p.add_argument("--journal")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("demo", help="offline end-to-end demo on synthetic data")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_demo)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
