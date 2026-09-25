# trading-agent: a Kalman-filter VWAP execution agent

An AI-operable trading agent that implements **Chen, Feng & Palomar, *Forecasting
Intraday Trading Volume: A Kalman Filter Approach*** (SSRN 3101695). It forecasts
15-minute volume with the paper's state-space model and uses the forecasts to
slice a parent order so the fill tracks the day's VWAP.

> **What this strategy is, and is not.** The paper is an *execution* strategy. It
> answers "I need to buy 50,000 SPY today. How much should I buy in each
> 15-minute bin so my average price matches VWAP?" It does not generate
> buy/sell signals or predict price direction. You choose the order; the agent
> works it.

## How it maps to the paper

| Paper | Code |
|---|---|
| log-volume decomposition `y = η + φ + μ + v` (Eq. 3), state space (Eqs. 4-5) | `kvwap/model.py` (`KalmanParams`) |
| Kalman filter, Algorithm 1; static/dynamic prediction (Eq. 9) | `kalman_filter`, `forecast_log_volume`, `advance` |
| Kalman smoother, Algorithm 2 + lag-one covariance (A.20-A.22) | `rts_smoother` |
| EM with closed-form M-step, Algorithm 3 (Eqs. 17-24) | `fit_em`, `m_step` |
| Robust Lasso filter (Eqs. 29-34) and its EM (Eqs. 35-36) | `kalman_filter(lam=...)` |
| MAPE (37), relative improvement (38) | `kvwap/vwap.py` |
| static VWAP (40), dynamic VWAP (41), tracking error (42) | `kvwap/vwap.py`, `kvwap/agent.py` |
| rolling out-of-sample test, cross-validation of N and λ (Sec. 4.1) | `kvwap/evaluate.py` |

**Where the code departs from the paper, and why:**

- **Robust threshold scaling.** Eq. 34 uses the threshold `λ/(2W) = λS/2`. With a
  fixed λ inside EM, that threshold shrinks faster (∝S) than the noise (∝√S).
  On contaminated data this tips EM into a degenerate solution: `r → 0` and
  every bin flagged as an outlier (reproduced in development). The default
  `lam_mode="sigma"` uses the threshold `λ√S`: the same Lasso problem with a
  per-bin λ, where λ now reads as "how many predictive standard deviations
  count as an outlier". `lam_mode="paper"` gives Eq. 34 verbatim.
- **Daily re-calibration.** Parameters are re-estimated once per day, not once
  per bin. The filter still updates every bin.
- **Lag-one smoother covariance.** It uses the identity `Σ_{τ,τ-1|N} = Σ_{τ|N} L'_{τ-1}`,
  which is algebraically the same as A.20-A.21. A test checks it against exact
  Gaussian conditioning.
- **CMEM benchmark.** Not implemented. The rolling-means (RM) benchmark is.

## Quick start

```bash
pip install -e '.[dev,yahoo]'
pytest -q                     # 43 offline tests
kvwap demo                    # offline end-to-end run on synthetic data
```

With real data (free):

```bash
kvwap fetch --source yahoo --symbol SPY --out data/SPY.csv            # no key needed
kvwap backtest --csv data/SPY.csv --cv-days 10 --json data/SPY.json   # validate + pick train_days, lam
cp config/example.toml my.toml   # set symbol/side/quantity and the CV'd train_days / lam
kvwap forecast --config my.toml                                        # today's volume curve + schedule
kvwap simulate --config my.toml --csv data/SPY.csv --date 2026-09-24   # replay a past day
kvwap run --config my.toml --dry-run                                   # live timing, no orders sent
# then set [broker] name = "alpaca-paper" to paper trade
```

## Free connectors

| Purpose | Connector | Key |
|---|---|---|
| Intraday bars | Yahoo Finance (`yfinance`), last ~60 days of 15m bars | none |
| Intraday bars | Alpaca Market Data: `iex` feed real-time, `sip` feed 15 min delayed on the free plan | free paper account (`APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`) |
| Intraday bars | Alpha Vantage `TIME_SERIES_INTRADAY` (some free keys don't include intraday) | `ALPHAVANTAGE_API_KEY` |
| Orders | Alpaca paper trading | same free Alpaca keys |
| Claude tools | Alpha Vantage MCP server (`.mcp.json`) | `ALPHAVANTAGE_API_KEY` |

Train and trade on the same feed. On Alpaca's IEX feed, set `risk.volume_scale ≈ 40`
so the participation cap refers to consolidated volume. With the delayed SIP feed
the agent simply runs one bin behind: missing bins skip the Kalman correction
step, as in paper Sec. 2.2.

## The agent

Each trading day, `kvwap run`:

1. calibrates the robust Kalman model by EM on the last `train_days` complete
   days (half days are excluded, as in the paper);
2. at every bin, filters today's finished bars and forecasts the remaining
   bins. It sizes the child order as `remaining × f_i / Σ_{j≥i} f_j`
   (dynamic, Eq. 41) or from the opening schedule (static, Eq. 40);
3. applies risk caps (participation of forecast volume, per-order and parent
   notional, `STOP` kill-switch file) and sends market orders, optionally
   split into slices across the bin;
4. after the close, journals slippage vs. market VWAP, completion, and
   forecast MAPE to `journal/executions.jsonl`.

Live orders need `broker.name = "alpaca-live"` **and** `KVWAP_ALLOW_LIVE=yes`.

## Claude Code skills

`.claude/skills/` makes Claude Code the operator of this agent:

- `kalman-vwap-agent`: the workflow and safety rules for this repo;
- vendored from [agiprolabs/claude-trading-skills](https://github.com/agiprolabs/claude-trading-skills)
  (MIT, see `.claude/skills/THIRD_PARTY_LICENSE.md`): `market-microstructure-traditional`,
  `slippage-modeling`, `risk-management`, `position-sizing`, `walk-forward-validation`,
  `trade-journal`, `ohlcv-processing`. Refresh them with `scripts/install_skills.sh`.

Example prompts: *"Backtest the Kalman VWAP model on AAPL and tell me whether it
beats rolling means"*, *"Plan a 20k-share sell of QQQ for tomorrow and rehearse
it on yesterday's data"*, *"Review today's execution journal"*.

## Results on synthetic data

These numbers come from `kvwap demo`, with data drawn from the paper's own model.
This is not market data.

| model | MAPE (clean) | VWAP TE bps (clean) | MAPE (10% outliers) |
|---|---|---|---|
| rolling means | 0.64 | 4.20 | 1.00 |
| Kalman static | 0.36 | 3.82 | 0.39 |
| Kalman dynamic | 0.26 | 3.13 | 0.33 |
| robust Kalman dynamic | 0.26 | 3.13 | **0.29** |

The ranking matches the paper: dynamic beats static, which beats RM, and the
robust filter matters only when the data are dirty. Check your own symbols
with `kvwap backtest` before trading them.

*Research and educational software. Not financial advice.*
