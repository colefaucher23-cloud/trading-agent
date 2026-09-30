"""Apply each strategy's static universe rules (strategies.toml) to energy_universe.csv.

Prints the names eligible today, using the 2026-09-30 snapshot's market cap and price as a
stand-in for the point-in-time screens. The ADV and history screens need price history and
are NOT applied here, so these lists are upper bounds for the backtest's universe on the
last date.

Run: python3 research/energy-strategies/select_universe.py [--csv out.csv]
"""
import csv
import pathlib
import sys
import tomllib
from collections import Counter

HERE = pathlib.Path(__file__).parent
CFG = tomllib.loads((HERE / "strategies.toml").read_text())
ROWS = list(csv.DictReader((HERE.parent / "energy-universe" / "energy_universe.csv").open()))


def num(x):
    return float(x) if x not in ("", None) else None


def eligible(cfg):
    out = []
    for r in ROWS:
        if r["asset_class"] not in cfg["asset_classes"] or r["alignment"] not in cfg["alignment"]:
            continue
        if r["security_type"] not in cfg["security_types"] or r["sub_industry"] in cfg.get("exclude_sub_industries", []):
            continue
        if cfg.get("exclude_foreign_primary") and r["foreign_primary"] == "yes":
            continue
        mc, px = num(r["market_cap_usd"]), num(r["last_price"])
        if mc is None or mc < cfg["min_market_cap_usd"] or px is None or px < cfg["min_price"]:
            continue
        out.append(r)
    return out


rows_out = []
for key in ("s1_residual_reversal", "s2_capital_discipline", "s4_earnings_drift"):
    cfg = CFG[key]
    names = eligible(cfg)
    print(f"\n{cfg['name']} [{key}]: {len(names)} names pass static + snapshot screens")
    for sub, n in Counter(r["sub_industry"] for r in names).most_common():
        print(f"  {n:>3}  {sub}")
    if key == "s1_residual_reversal":
        shorts = [r for r in names if num(r["market_cap_usd"]) >= cfg["short_min_market_cap_usd"]]
        print(f"  short-eligible (market cap >= ${cfg['short_min_market_cap_usd'] / 1e9:.0f}B): {len(shorts)}")
    rows_out += [{"strategy": key, "symbol": r["symbol"], "sub_industry": r["sub_industry"],
                  "market_cap_usd": r["market_cap_usd"]} for r in names]

s3 = CFG["s3_trend_carry"]
etps = {r["symbol"]: r for r in ROWS if r["asset_class"] == "etp"}
missing = [s for s in s3["instruments"] + list(s3["carry_proxy"].values()) if s not in etps]
print(f"\n{s3['name']} [s3_trend_carry]: {len(s3['instruments'])} instruments "
      f"({', '.join(s3['instruments'])}); carry proxies {sorted(set(s3['carry_proxy'].values()))}; "
      f"missing from universe: {missing or 'none'}")
rows_out += [{"strategy": "s3_trend_carry", "symbol": s, "sub_industry": etps[s]["etp_category"], "market_cap_usd": ""}
             for s in s3["instruments"]]

if "--csv" in sys.argv:
    path = pathlib.Path(sys.argv[sys.argv.index("--csv") + 1])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {path}")
