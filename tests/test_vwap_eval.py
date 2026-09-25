import numpy as np
import pytest

from kvwap.data import DailyGrid, bars_to_grid, default_true_params, simulate_log_volume, synthetic_bars
from kvwap.evaluate import BacktestConfig, cross_validate, rolling_backtest
from kvwap.vwap import dynamic_weight, improvement, mape, replicated_vwap, slippage_bps, static_weights, tracking_error_bps, vwap


def test_static_weights_sum_to_one():
    w = static_weights([1.0, 2.0, 3.0, 4.0])
    assert w.sum() == pytest.approx(1.0)
    assert w[3] == pytest.approx(0.4)


def test_dynamic_weights_with_perfect_forecast_replicate_vwap():
    rng = np.random.default_rng(0)
    vol = rng.uniform(1, 10, 26)
    price = 100 + rng.normal(0, 1, 26).cumsum()
    executed, w = 0.0, []
    for i in range(26):
        wi = dynamic_weight(vol[i:], executed)
        w.append(wi)
        executed += wi
    assert sum(w) == pytest.approx(1.0)
    np.testing.assert_allclose(w, vol / vol.sum())
    assert tracking_error_bps(vwap(price, vol), replicated_vwap(price, w)) == pytest.approx(0.0, abs=1e-9)


def test_last_bin_takes_remainder():
    assert dynamic_weight([5.0], 0.7) == pytest.approx(0.3)


def test_metrics():
    assert mape([100, 200], [110, 180]) == pytest.approx(0.1)
    assert improvement(1.0, 0.36) == pytest.approx(64.0)
    assert slippage_bps(101.0, 100.0, "buy") == pytest.approx(100.0)
    assert slippage_bps(101.0, 100.0, "sell") == pytest.approx(-100.0)


@pytest.fixture(scope="module")
def grid():
    return bars_to_grid(synthetic_bars(n_days=70, seed=4))


def test_backtest_ranks_models_like_the_paper(grid):
    res = rolling_backtest(grid, BacktestConfig(train_days=40), start=50)
    s = res.summary()["mape"]
    assert s["kf_dynamic"]["mean"] < s["kf_static"]["mean"] < s["rm_static"]["mean"]
    assert s["rkf_dynamic"]["mean"] < s["rm_static"]["mean"]
    assert len(res.dates) == 20


def test_robust_filter_wins_on_contaminated_data(grid):
    clean, dirty = simulate_log_volume(default_true_params(26), grid.n_days, seed=4, outlier_frac=0.1, outlier_scale=3.0)
    g = DailyGrid(grid.dates, np.exp(clean), grid.price, grid.session)
    res = rolling_backtest(g, BacktestConfig(train_days=40, lam=2.0), start=55, observed_log=dirty)
    s = res.summary()["mape"]
    assert s["rkf_dynamic"]["mean"] < s["kf_dynamic"]["mean"]


def test_cross_validation_picks_from_grid(grid):
    cv = cross_validate(grid, cv_days=5, test_start=60, train_days_grid=(20, 40), lam_grid=(None, 3.0))
    assert cv.best_train_days in (20, 40)
    assert len(cv.scores) == 4
