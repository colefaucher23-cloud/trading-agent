---
name: kalman-vwap-agent
description: Operate the Kalman-filter intraday volume forecaster and VWAP execution agent in this repo (Chen, Feng & Palomar). Use when asked to forecast intraday volume, backtest or cross-validate the model, plan or run a VWAP order, replay a day, or review execution quality from the journal.
---

# Kalman VWAP Agent

This repo implements the paper *Forecasting Intraday Trading Volume: A Kalman
Filter Approach* end to end: log-volume state-space model, closed-form EM,
Lasso-robust filtering, and static/dynamic VWAP replication, wrapped in an
execution agent. **It is an execution strategy, not an alpha strategy.** It
decides *how* to slice an order the user already wants to trade so the fill
tracks the day's VWAP; it never decides *what* or *whether* to trade.

## Workflow

1. **Get data** (free):
   `kvwap fetch --source yahoo --symbol SPY --out data/SPY.csv` (no key, ~60 days)
   or `--source alpaca --feed sip` (free paper keys, years of history, 15-min delayed).
2. **Validate on this symbol before trading it**:
   `kvwap backtest --csv data/SPY.csv --cv-days 10 --json data/SPY_bt.json`.
   Cross-validation picks `train_days` and `lam` by dynamic MAPE. Expect
   `rkf_dynamic` < `kf_static` < `rm_static` on MAPE; if the Kalman models do
   not beat RM, say so and do not recommend running live.
3. **Plan**: copy `config/example.toml`, set order + the CV'd `train_days`/`lam`,
   then `kvwap forecast --config my.toml` prints today's volume curve and schedule.
4. **Rehearse**: `kvwap simulate --config my.toml --csv data/SPY.csv --date YYYY-MM-DD`.
5. **Run**: `kvwap run --config my.toml --dry-run` first; then `broker.name =
   "alpaca-paper"`. Live trading requires `alpaca-live` **and**
   `KVWAP_ALLOW_LIVE=yes`; never set either without the user's explicit instruction.
6. **Review**: read `journal/executions.jsonl` (`calibrated`, `bin`, `order`,
   `summary` events). Report slippage vs market VWAP, completion %, static vs
   dynamic forecast MAPE, and any `capped_by` risk events.

## Parameter guidance

| param | meaning | default / advice |
|---|---|---|
| `mode` | `dynamic` = Eq. 41 re-plans every bin; `static` = Eq. 40 fixed at open | dynamic (paper's best) |
| `lam` | robust filter outlier threshold, in predictive std devs | 2-4 via CV; `none` = standard KF |
| `lam_mode` | `sigma` (stable) or `paper` (threshold = lam*S/2, Eq. 34 verbatim) | sigma; `paper` can collapse EM (r -> 0) |
| `train_days` | EM calibration window | 20-60 via CV |
| `max_participation` | child <= share of forecast bin volume | 5-15%; set `volume_scale`~40 on IEX feed |
| `slices_per_bin` | TWAP slices inside each 15-min bin | 1-5 |

## Safety rules

- Always run `pytest -q` after changing `kvwap/`.
- Never loosen risk limits, enable `final_sweep`, or switch to a live broker unless the user explicitly asks.
- `touch STOP` in the working dir halts the agent's new orders immediately.
- Half days are excluded (the paper does the same); `kvwap run` refuses them when Alpaca's calendar is available.
- Nothing here is financial advice; say so when presenting results.

## Related skills in this repo

`market-microstructure-traditional` (VWAP/IS benchmarks), `slippage-modeling`
(choose `--cost-bps`), `risk-management` and `position-sizing` (parent order
limits), `walk-forward-validation` (the rolling backtest follows it),
`trade-journal` (journal review), `ohlcv-processing` (bar cleaning).
