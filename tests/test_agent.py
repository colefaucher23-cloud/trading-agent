import os

import pandas as pd
import pytest

from kvwap.agent import AgentConfig, ParentOrder, SimClock, VWAPExecutionAgent, simulate_day
from kvwap.broker import DryRunBroker
from kvwap.config import load_config
from kvwap.connectors import ReplaySource
from kvwap.data import Session, synthetic_bars
from kvwap.journal import Journal
from kvwap.risk import RiskLimits, clamp_child


@pytest.fixture(scope="module")
def bars():
    return synthetic_bars(n_days=45, seed=9)


@pytest.fixture(scope="module")
def last_day(bars):
    return sorted(set(bars.index.tz_convert("America/New_York").date))[-1]


def cfg(**kw):
    base = dict(order=ParentOrder("SYN", "buy", 50_000), train_days=30, risk=RiskLimits(max_participation=0.5))
    base.update(kw)
    return AgentConfig(**base)


@pytest.mark.parametrize("mode", ["dynamic", "static"])
def test_replay_completes_order(bars, last_day, mode):
    j = Journal(echo=False)
    s = simulate_day(cfg(mode=mode, slices_per_bin=2), bars, last_day, cost_bps=1.0, journal=j)
    assert s["filled"] == 50_000
    assert s["orders"] == 52
    assert s["slippage_vs_vwap_bps"] is not None
    assert s["dynamic_forecast_mape"] < s["static_forecast_mape"] + 0.2
    kinds = [e["event"] for e in j.events]
    assert kinds[0] == "calibrated" and kinds[-1] == "summary"
    # no look-ahead: the calibration window ends the day before
    assert j.events[0]["last_train_day"] < str(last_day)


def test_agent_only_sees_finished_bins(bars, last_day):
    j = Journal(echo=False)
    simulate_day(cfg(), bars, last_day, journal=j)
    for e in (e for e in j.events if e["event"] == "bin"):
        assert e["observed_bins"] == e["bin"]


def test_delayed_feed_still_completes(bars, last_day):
    j = Journal(echo=False)
    s = simulate_day(cfg(), bars, last_day, journal=j, feed_delay_minutes=15)
    assert s["filled"] == 50_000
    obs = [e["observed_bins"] for e in j.events if e["event"] == "bin"]
    assert obs[0] == 0 and obs[5] == 4  # one bin behind


def test_participation_cap_limits_children(bars, last_day):
    j = Journal(echo=False)
    c = cfg(order=ParentOrder("SYN", "sell", 5_000_000), risk=RiskLimits(max_participation=0.05))
    s = simulate_day(c, bars, last_day, journal=j)
    assert s["filled"] < 5_000_000
    for e in (e for e in j.events if e["event"] == "bin"):
        assert e["send"] <= 0.05 * e["forecast_volume"] + 1


def test_final_sweep_completes(bars, last_day):
    c = cfg(order=ParentOrder("SYN", "buy", 2_000_000), risk=RiskLimits(max_participation=0.05, final_sweep=True))
    s = simulate_day(c, bars, last_day, journal=Journal(echo=False))
    assert s["filled"] == 2_000_000


def test_kill_switch_stops_trading(bars, last_day, tmp_path):
    stop = tmp_path / "STOP"
    stop.write_text("halt")
    c = cfg(risk=RiskLimits(max_participation=0.5, kill_switch_file=str(stop)))
    j = Journal(echo=False)
    s = simulate_day(c, bars, last_day, journal=j)
    assert s["filled"] == 0
    assert any(e["event"] == "kill_switch" for e in j.events)


def test_dry_run_sends_nothing_but_plans_full_quantity(bars, last_day):
    session = Session()
    clock = SimClock(session.bin_start_ts(last_day, 0) - pd.Timedelta(minutes=1))
    src = ReplaySource(bars, clock.now)
    agent = VWAPExecutionAgent(cfg(), src, DryRunBroker(), clock, Journal(echo=False))
    s = agent.run_day(last_day)
    assert s["filled"] == 0
    assert agent.committed_qty() == 50_000


def test_clamp_child():
    lim = RiskLimits(max_participation=0.1, max_child_notional=10_000)
    d = clamp_child(5_000, forecast_bin_volume=20_000, ref_price=10.0, limits=lim)
    assert d.qty == 1_000 and len(d.reasons) == 2
    assert clamp_child(5_000, 20_000, 10.0, RiskLimits(max_participation=0.1, final_sweep=True), last_bin=True).qty == 5_000


def test_parent_notional_limit(bars, last_day):
    c = cfg(risk=RiskLimits(max_parent_notional=1_000.0))
    with pytest.raises(ValueError, match="max_parent_notional"):
        simulate_day(c, bars, last_day, journal=Journal(echo=False))


def test_example_config_loads():
    here = os.path.dirname(__file__)
    rc = load_config(os.path.join(here, "..", "config", "example.toml"))
    assert rc.agent.order.symbol == "SPY"
    assert rc.broker == "dry-run"
    assert rc.agent.session.bins_per_day == 26
