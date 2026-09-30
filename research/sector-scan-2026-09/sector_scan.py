"""Sector relative-strength scan (data through 2026-09-29 close).

Reads the Alpha Vantage ANALYTICS_FIXED_WINDOW snapshot in data/av_fixed_window_raw.json
and produces the relative-strength tables used in REPORT.md:

  results/core_windows.csv      returns + excess-vs-SPY for the 11 sectors and benchmarks, 1-6M and 12M
  results/monthly_path.csv      month-by-month path reconstructed from the nested windows
  results/sector_scores.csv     composite relative-strength score for the 11 SPDR sectors
  results/subindustry.csv       1/3/6M returns, excess vs SPY and vol for sub-industry ETFs

All returns are price returns on daily closes (dividends excluded). Every window ends on
2026-09-29; the "Nmonth" window starts on the first trading day on/after the same calendar
day N months earlier (6M = 2026-03-30, 3M = 2026-06-29, 1M = 2026-08-31, ...).

Run: python3 research/sector-scan-2026-09/sector_scan.py
"""
import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
RAW = json.loads((HERE / "data" / "av_fixed_window_raw.json").read_text())["windows"]
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

BENCH = "SPY"
SECTORS = {
    "XLK": "Technology", "XLV": "Health Care", "XLE": "Energy", "XLF": "Financials",
    "XLC": "Communication Svcs", "XLI": "Industrials", "XLY": "Consumer Discretionary",
    "XLRE": "Real Estate", "XLP": "Consumer Staples", "XLB": "Materials", "XLU": "Utilities",
}
BENCHMARKS = {"SPY": "S&P 500", "RSP": "S&P 500 equal-weight", "QQQ": "Nasdaq-100", "IWM": "Russell 2000"}
# Invesco S&P 500 equal-weight sector ETFs, used to judge breadth inside each sector.
EQUAL_WEIGHT = {
    "XLK": "RSPT", "XLV": "RSPH", "XLE": "RSPG", "XLF": "RSPF", "XLC": "RSPC", "XLI": "RSPN",
    "XLY": "RSPD", "XLRE": "RSPR", "XLP": "RSPS", "XLB": "RSPM", "XLU": "RSPU",
}
SUBINDUSTRY = {
    "SMH": ("Technology", "Semiconductors"), "IGV": ("Technology", "Software"),
    "MAGS": ("Mega-cap", "Magnificent 7"),
    "IBB": ("Health Care", "Biotech (large-cap)"), "XBI": ("Health Care", "Biotech (equal-wt, SMID)"),
    "XPH": ("Health Care", "Pharma (equal-wt)"), "IHF": ("Health Care", "Providers / managed care"),
    "IHI": ("Health Care", "Medical devices"),
    "XOP": ("Energy", "Exploration & production"), "OIH": ("Energy", "Oil services"),
    "KRE": ("Financials", "Regional banks"), "IAI": ("Financials", "Brokers & exchanges"),
    "KIE": ("Financials", "Insurance"),
    "ITA": ("Industrials", "Aerospace & defense"), "IYT": ("Industrials", "Transports"),
    "JETS": ("Industrials", "Airlines"), "GRID": ("Industrials", "Grid infrastructure"),
    "GDX": ("Materials", "Gold miners"), "COPX": ("Materials", "Copper miners"),
    "XME": ("Materials", "Metals & mining"), "URA": ("Energy/Utilities", "Uranium & nuclear"),
    "TAN": ("Utilities/Energy", "Solar"),
    "XHB": ("Consumer Discretionary", "Homebuilders"), "XRT": ("Consumer Discretionary", "Retail (equal-wt)"),
}
MONTH_WINDOWS = ["6month", "5month", "4month", "3month", "2month", "1month"]


def ret(window, sym):
    return RAW[window]["cumulative_return"].get(sym, np.nan)


def vol(window, sym):
    return RAW[window].get("stddev_annualized", {}).get(sym, np.nan)


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


# ---------------------------------------------------------------- core windows
core = list(SECTORS) + list(BENCHMARKS)
rows = []
for sym in core:
    row = {"symbol": sym, "name": SECTORS.get(sym) or BENCHMARKS[sym]}
    for w in MONTH_WINDOWS + ["1year"]:
        row[f"ret_{w}"] = ret(w, sym)
        row[f"xs_{w}"] = ret(w, sym) - ret(w, BENCH)
    rows.append(row)
core_df = pd.DataFrame(rows).set_index("symbol")
core_df.to_csv(OUT / "core_windows.csv", float_format="%.4f")

# ---------------------------------------------------------------- monthly path
# Every window ends on the same day, so the level at each window start relative to the
# end is 1 / (1 + R_window). Chaining those gives a month-by-month path.
anchors_sorted = [(RAW[w]["min_dt"], w) for w in MONTH_WINDOWS] + [(RAW["1month"]["max_dt"], None)]
path_rows, seg_rows = [], []
for sym in core:
    levels = [1.0 / (1.0 + ret(w, sym)) if w else 1.0 for _, w in anchors_sorted]
    base = levels[0]
    path_rows.append({"symbol": sym, **{d: 100 * lv / base for (d, _), lv in zip(anchors_sorted, levels)}})
    seg = {"symbol": sym}
    for (d0, _), (d1, _), l0, l1 in zip(anchors_sorted, anchors_sorted[1:], levels, levels[1:]):
        seg[f"{d0}->{d1}"] = l1 / l0 - 1
    seg_rows.append(seg)
path_df = pd.DataFrame(path_rows).set_index("symbol")
seg_df = pd.DataFrame(seg_rows).set_index("symbol")
seg_xs = seg_df.sub(seg_df.loc[BENCH], axis=1)
path_df.to_csv(OUT / "monthly_path.csv", float_format="%.2f")

# ---------------------------------------------------------------- sector composite score
sec = core_df.loc[list(SECTORS)].copy()
xs_cols = [f"xs_{w}" for w in MONTH_WINDOWS]
# 1. Multi-horizon relative strength: average excess return over the 1..6 month windows.
#    Averaging over six start dates damps the start-date sensitivity of any single window
#    (the 6M window starts at the late-March war low, which flatters high-beta sectors).
sec["mh_rs"] = sec[xs_cols].mean(axis=1)
# 2. Consistency: number of the six monthly segments in which the sector beat SPY.
sec["months_beat_spy"] = (seg_xs.loc[list(SECTORS)] > 0).sum(axis=1)
# 3. Acceleration: last-3M excess minus prior-3M excess (positive = strengthening).
prior3 = (1 + sec["ret_6month"]) / (1 + sec["ret_3month"]) - 1
prior3_spy = (1 + ret("6month", BENCH)) / (1 + ret("3month", BENCH)) - 1
sec["accel"] = sec["xs_3month"] - (prior3 - prior3_spy)
# 4. Breadth: 3M return of the equal-weight version of the sector minus RSP's 3M return.
#    Positive means the typical stock in the sector is beating the typical S&P 500 stock,
#    not only the sector's mega-caps.
sec["ew_ticker"] = [EQUAL_WEIGHT[s] for s in sec.index]
sec["ew_breadth_3m"] = [ret("3month", EQUAL_WEIGHT[s]) - ret("3month", "RSP") for s in sec.index]
sec["ew_breadth_6m"] = [ret("6month", EQUAL_WEIGHT[s]) - ret("6month", "RSP") for s in sec.index]
# 5. Risk-adjusted 3M excess: 3M excess return scaled by the sector's 3M annualized vol.
sec["vol_3m"] = [vol("3month", s) for s in sec.index]
sec["xs3_per_vol"] = sec["xs_3month"] / sec["vol_3m"]

WEIGHTS = {"mh_rs": 0.35, "months_beat_spy": 0.15, "accel": 0.15, "ew_breadth_3m": 0.20, "xs3_per_vol": 0.15}
sec["score"] = sum(w * zscore(sec[c]) for c, w in WEIGHTS.items())
sec = sec.sort_values("score", ascending=False)
sec["rank"] = range(1, len(sec) + 1)
sec.to_csv(OUT / "sector_scores.csv", float_format="%.4f")

# Weight sensitivity: re-rank under alternative weightings to see which conclusions are robust.
ALT_WEIGHTS = {
    "base": WEIGHTS,
    "equal": {c: 0.2 for c in WEIGHTS},
    "pure momentum (avg excess only)": {"mh_rs": 1.0},
    "no acceleration": {"mh_rs": 0.45, "months_beat_spy": 0.2, "ew_breadth_3m": 0.2, "xs3_per_vol": 0.15},
    "no breadth": {"mh_rs": 0.45, "months_beat_spy": 0.2, "accel": 0.2, "xs3_per_vol": 0.15},
    "recent-heavy (3M excess/vol + accel)": {"xs3_per_vol": 0.5, "accel": 0.5},
}
sens = pd.DataFrame({
    name: sum(w * zscore(sec[c]) for c, w in wts.items()).rank(ascending=False).astype(int)
    for name, wts in ALT_WEIGHTS.items()
})
sens.to_csv(OUT / "weight_sensitivity.csv")

# ---------------------------------------------------------------- sub-industries
sub_rows = []
for sym, (parent, label) in SUBINDUSTRY.items():
    r1, r3, r6 = ret("1month", sym), ret("3month", sym), ret("6month", sym)
    sub_rows.append({
        "symbol": sym, "parent": parent, "industry": label,
        "ret_1m": r1, "ret_3m": r3, "ret_6m": r6,
        "xs_1m": r1 - ret("1month", BENCH), "xs_3m": r3 - ret("3month", BENCH), "xs_6m": r6 - ret("6month", BENCH),
        "vol_6m": vol("6month", sym),
    })
sub = pd.DataFrame(sub_rows).set_index("symbol")
sub["mh_rs"] = sub[["xs_1m", "xs_3m", "xs_6m"]].mean(axis=1)
sub = sub.sort_values("mh_rs", ascending=False)
sub.to_csv(OUT / "subindustry.csv", float_format="%.4f")


# ---------------------------------------------------------------- console / markdown output
def pct(x, signed=True):
    return "n/a" if pd.isna(x) else (f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%")


def pp(x):
    return "n/a" if pd.isna(x) else f"{x * 100:+.1f}"


print("## Composite sector score (11 SPDR sectors)\n")
print("| Rank | Sector | ETF | 1M | 3M | 6M | 12M | Avg excess 1-6M (pp) | Months beat SPY (of 6) | Accel (pp) | EW breadth 3M (pp) | 3M excess / vol | Score |")
print("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for s, r in sec.iterrows():
    print(f"| {r['rank']} | {r['name']} | {s} | {pct(r.ret_1month)} | {pct(r.ret_3month)} | {pct(r.ret_6month)} | {pct(r.ret_1year)} "
          f"| {pp(r.mh_rs)} | {int(r.months_beat_spy)} | {pp(r.accel)} | {pp(r.ew_breadth_3m)} | {r.xs3_per_vol:+.2f} | {r.score:+.2f} |")
print("\n## Rank under alternative weightings\n")
print("| Sector | " + " | ".join(sens.columns) + " |")
print("|---|" + "---:|" * len(sens.columns))
for s in sec.index:
    print(f"| {SECTORS[s]} | " + " | ".join(str(v) for v in sens.loc[s]) + " |")
print(f"\nBenchmarks: " + ", ".join(
    f"{b} 1M {pct(ret('1month', b))} / 3M {pct(ret('3month', b))} / 6M {pct(ret('6month', b))} / 12M {pct(ret('1year', b))}"
    for b in BENCHMARKS))

print("\n## Monthly excess return vs SPY (pp)\n")
cols = list(seg_xs.columns)
short = [c.split("->")[0][5:] + "→" + c.split("->")[1][5:] for c in cols]
print("| ETF | " + " | ".join(short) + " |")
print("|---|" + "---:|" * len(cols))
for s in list(sec.index) + ["RSP", "QQQ", "IWM"]:
    print(f"| {s} | " + " | ".join(pp(seg_xs.loc[s, c]) for c in cols) + " |")
print("\nSPY monthly return: " + ", ".join(f"{k}: {pct(seg_df.loc['SPY', c])}" for k, c in zip(short, cols)))

print("\n## Cap-weight vs equal-weight inside each sector\n")
print("| Sector | CW 1M | EW 1M | CW 3M | EW 3M | CW 6M | EW 6M |")
print("|---|---:|---:|---:|---:|---:|---:|")
for s in sec.index:
    e = EQUAL_WEIGHT[s]
    print(f"| {SECTORS[s]} ({s}/{e}) | {pct(ret('1month', s))} | {pct(ret('1month', e))} | {pct(ret('3month', s))} | {pct(ret('3month', e))} "
          f"| {pct(ret('6month', s))} | {pct(ret('6month', e))} |")
print(f"| S&P 500 (SPY/RSP) | {pct(ret('1month', 'SPY'))} | {pct(ret('1month', 'RSP'))} | {pct(ret('3month', 'SPY'))} | {pct(ret('3month', 'RSP'))} "
      f"| {pct(ret('6month', 'SPY'))} | {pct(ret('6month', 'RSP'))} |")

print("\n## Sub-industries ranked by average excess return (1M/3M/6M)\n")
print("| ETF | Group | Industry | 1M | 3M | 6M | Avg excess (pp) | 6M vol |")
print("|---|---|---|---:|---:|---:|---:|---:|")
for s, r in sub.iterrows():
    print(f"| {s} | {r.parent} | {r.industry} | {pct(r.ret_1m)} | {pct(r.ret_3m)} | {pct(r.ret_6m)} | {pp(r.mh_rs)} | {pct(r.vol_6m, signed=False)} |")
