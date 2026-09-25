import numpy as np
import pandas as pd

from kvwap.data import Session, bars_to_grid, detect_right_labeled, intraday_bins, synthetic_bars


def test_session_bins():
    s = Session()
    assert s.bins_per_day == 26
    assert s.bin_starts()[0].strftime("%H:%M") == "09:30"
    assert s.bin_starts()[-1].strftime("%H:%M") == "15:45"
    assert Session(tz="Asia/Tokyo", open="09:00", close="11:30").bins_per_day == 10


def test_grid_from_synthetic_bars():
    bars = synthetic_bars(n_days=5)
    g = bars_to_grid(bars)
    assert g.volume.shape == (5, 26)
    assert (g.volume > 0).all()


def test_half_days_and_gaps_are_dropped():
    bars = synthetic_bars(n_days=4)
    local = bars.index.tz_convert("America/New_York")
    # day 2 closes at 13:00 (half day); day 3 misses one bar
    d2, d3 = sorted(set(local.date))[1:3]
    half = (local.date == d2) & (local.strftime("%H:%M") >= "13:00")
    gap = (local.date == d3) & (local.strftime("%H:%M") == "11:00")
    g = bars_to_grid(bars[~(half | gap)])
    assert g.n_days == 2
    assert d2 not in g.dates and d3 not in g.dates


def test_one_minute_bars_are_aggregated():
    idx = pd.date_range("2026-03-02 09:30", "2026-03-02 15:59", freq="1min", tz="America/New_York")
    bars = pd.DataFrame({"close": np.linspace(100, 101, len(idx)), "volume": 10.0}, index=idx)
    g = bars_to_grid(bars)
    assert g.volume.shape == (1, 26)
    np.testing.assert_allclose(g.volume, 150.0)


def test_right_labeled_bars_detected_and_shifted():
    bars = synthetic_bars(n_days=3)
    shifted = bars.copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=15)  # label by bar end: 09:45 .. 16:00
    local = shifted.index.tz_convert("America/New_York")
    assert detect_right_labeled(local, Session())
    np.testing.assert_allclose(bars_to_grid(shifted).volume, bars_to_grid(bars).volume)


def test_zero_volume_is_floored():
    bars = synthetic_bars(n_days=2)
    bars.iloc[3, bars.columns.get_loc("volume")] = 0.0
    g = bars_to_grid(bars)
    assert (g.volume > 0).all()


def test_intraday_bins_partial_day():
    bars = synthetic_bars(n_days=2)
    day = sorted(set(bars.index.tz_convert("America/New_York").date))[-1]
    partial = bars[bars.index < Session().bin_start_ts(day, 5)]
    vol, px, vw = intraday_bins(partial, Session(), day)
    assert np.isfinite(vol[:5]).all() and np.isnan(vol[5:]).all()
    assert np.isnan(vw).all()


def test_turnover_normalisation():
    bars = synthetic_bars(n_days=2)
    g = bars_to_grid(bars, shares_outstanding=1e6)
    np.testing.assert_allclose(g.volume * 1e6, bars_to_grid(bars).volume)
