"""Merge one Alpha Vantage ANALYTICS_FIXED_WINDOW payload into av_fixed_window_raw.json.

Usage: python3 add_window.py <window> '<json payload from the tool>'

Every calculation in the payload is stored under a normalized key
(CUMULATIVE_RETURN -> cumulative_return, STDDEV(ANNUALIZED=TRUE) -> stddev_annualized,
MAX -> max_daily_return, MIN -> min_daily_return; MAX/MIN are computed by the API on
daily returns, not on price levels).
"""
import json, sys, pathlib

KEYS = {
    "CUMULATIVE_RETURN": "cumulative_return",
    "STDDEV(ANNUALIZED=TRUE)": "stddev_annualized",
    "MAX": "max_daily_return",
    "MIN": "min_daily_return",
}

path = pathlib.Path(__file__).with_name("av_fixed_window_raw.json")
window, payload = sys.argv[1], json.loads(sys.argv[2])
data = json.loads(path.read_text())
w = data["windows"].setdefault(window, {})
w.setdefault("min_dt", payload["meta_data"]["min_dt"])
w.setdefault("max_dt", payload["meta_data"]["max_dt"])
assert (w["min_dt"], w["max_dt"]) == (payload["meta_data"]["min_dt"], payload["meta_data"]["max_dt"]), "window dates differ"
for calc, values in payload["payload"]["RETURNS_CALCULATIONS"].items():
    w.setdefault(KEYS[calc], {}).update(values)
path.write_text(json.dumps(data, indent=1))
print({k: len(v["cumulative_return"]) for k, v in data["windows"].items()})
