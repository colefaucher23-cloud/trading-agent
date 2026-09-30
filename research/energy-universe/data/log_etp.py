"""Append one Alpha Vantage ANALYTICS_FIXED_WINDOW result (or error) to etp_validation.jsonl.

Usage: python3 log_etp.py SYM1,SYM2,... '<tool output json>'
"""
import json, pathlib, re, sys

path = pathlib.Path(__file__).with_name("etp_validation.jsonl")
batch, raw = sys.argv[1].split(","), json.loads(sys.argv[2])
if "error" in raw:
    msg = raw["error"] if isinstance(raw["error"], str) else json.dumps(raw["error"])
    m = re.search(r"ticker (\S+) for the date range", msg)
    rec = {"batch": batch, "status": "error", "missing": m.group(1) if m else None, "error": msg[:200]}
else:
    calc = raw["payload"]["RETURNS_CALCULATIONS"]
    rec = {"batch": batch, "status": "ok", "min_dt": raw["meta_data"]["min_dt"], "max_dt": raw["meta_data"]["max_dt"],
           "ret_1m": calc["CUMULATIVE_RETURN"], "vol_1m": calc.get("STDDEV(ANNUALIZED=TRUE)", {})}
with path.open("a") as f:
    f.write(json.dumps(rec) + "\n")
print(rec["status"], rec.get("missing") or sorted(rec.get("ret_1m", {})))
