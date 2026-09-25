# trading-agent

Kalman-filter intraday volume forecasting + VWAP execution agent, implementing
Chen, Feng & Palomar, "Forecasting Intraday Trading Volume: A Kalman Filter
Approach" (SSRN 3101695). Package: `kvwap/`. Operating guide: `.claude/skills/kalman-vwap-agent/SKILL.md`.

## Commands
- Install: `pip install -e '.[dev,yahoo]'`
- Tests: `pytest -q` (offline, ~20 s)
- Offline demo: `kvwap demo`
- CLI: `kvwap {fetch,backtest,forecast,simulate,run} --help`

## Layout
- `kvwap/model.py` – state-space model, filter (Alg. 1 + robust Eqs. 31-34), RTS smoother (Alg. 2), EM (Alg. 3 / Eqs. 35-36), forecasts (Eq. 9)
- `kvwap/vwap.py` – static/dynamic VWAP weights (Eqs. 40-41), MAPE (37), tracking error (42)
- `kvwap/evaluate.py` – rolling out-of-sample backtest and CV (Sec. 4)
- `kvwap/agent.py` – live/replay execution agent; `risk.py`, `broker.py`, `connectors.py`, `journal.py`
- `.claude/skills/` – vendored skills from agiprolabs/claude-trading-skills (MIT) + `kalman-vwap-agent`

## Conventions
- Bins are 0-based in code (paper is 1-based); `tau` boundary = transition from bin I-1 to bin 0.
- Robust `lam` defaults to `lam_mode="sigma"` (threshold in predictive std devs); `"paper"` is Eq. 34 verbatim.
- Never enable live trading (`alpaca-live`, `KVWAP_ALLOW_LIVE=yes`) or loosen `[risk]` without an explicit user request.
