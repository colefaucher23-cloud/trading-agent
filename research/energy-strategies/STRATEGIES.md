# Four energy strategies (pre-registered specs for backtesting)

Status: design only. Nothing here has been backtested or fitted to data.

- **Numbering.** S1–S4 are labels only and imply no preference; the backtest will decide.
- **Parameters.** Machine-readable in [`strategies.toml`](strategies.toml).
- **Universe.** [`../energy-universe/UNIVERSE.md`](../energy-universe/UNIVERSE.md) covers 386 US-listed energy securities as of 2026-09-30:
  - 285 equities
  - 8 closed-end funds
  - 72 ETPs
  - 21 preferreds, notes and warrants
- **Eligibility.** [`select_universe.py`](select_universe.py) applies each strategy's static filters to that universe. Its output is [`eligible_2026-09-30.csv`](eligible_2026-09-30.csv).

| | S1 Residual Reversal | S2 Capital-Discipline Value | S3 Trend & Carry | S4 Commodity-Hedged Earnings Drift |
|---|---|---|---|---|
| Return source | Liquidity provision (fading non-fundamental flow) | Fundamental factor premium (investment, payout, value) | Time-series momentum + futures carry (hedging pressure) | Under-reaction to firm-specific news |
| Information used | Stock's own residual returns | Financial statements | ETP prices + futures-curve shape | Earnings events (surprise + announcement return) |
| Instruments | Single stocks + ETF hedges | Single stocks | 8 ETPs (5 equity, 3 commodity) | Single stocks + ETF hedges |
| Net exposure | Neutral to sector, oil and gas | Neutral within each sub-industry | Directional long/short | Hedged; idiosyncratic |
| Holding period | 5 trading days | ~1 quarter | Weeks to months | ≤ 40 trading days per event |
| Rebalance | Daily (5 overlapping sleeves) | Quarterly | Weekly | Event-driven, checked daily |
| Eligible today | 144 stocks (113 short-eligible)* | 157 stocks* | 8 ETPs (+ 2 carry proxies) | 165 stocks* |

\*The static filters plus the snapshot's market cap and price. The point-in-time ADV and history screens run inside the backtest and will shrink these counts.

---

## S1 — Energy Residual Reversal

The strategy fades short-term moves in liquid energy stocks that are not explained by the sector, the stock's peers, or the oil and gas futures strips.

### Universe and position sizing

**Eligible securities**

- Energy equities tagged `core` or `borderline`.
- Common shares, partnership units and LLC units.
- Not eligible:
  - ADRs and other foreign-primary listings (`foreign_primary = yes`: BP, TTE, E, EQNR, TS, Oslo/Copenhagen-primary tankers). Their home-market close creates stale-price "reversals".
  - Royalty trusts.
  - Gas-fired power solutions.

**Point-in-time screens at each date**

- Price ≥ $5.
- 60-day median daily dollar volume ≥ $20M.
- Market cap ≥ $1B.
- At least 252 trading days of history.
- **Short side only:** market cap ≥ $2B and borrow fee < 5%/yr.

**Position sizing**

- **Books.** A long book of 50% of NAV and a short book of 50% (gross 100%).
- **Sleeves.** The books are split across 5 overlapping daily sleeves of 1/5 of capital each.
- **Weights.** Within a sleeve, weight ∝ score ÷ residual volatility.
- **Name cap.** 4% of NAV per name, summed across sleeves.
- **Hedges.**
  - Net exposure to XLE, USL and UNL is kept at zero using those ETFs as hedges.
  - Net exposure to any one sub-industry is held within ±5% of NAV.

### Entry/exit logic and factors

**Factor model (per stock, daily)**

- Exponentially weighted OLS over 252 days, with a 126-day half-life.
- Four factors:
  - XLE.
  - A leave-one-out, equal-weighted average of the stock's sub-industry peers (needs ≥ 5 peers).
  - USL, the 12-month WTI strip.
  - UNL, the 12-month natural-gas strip.
- The strips are used because equity values key off the curve, not the prompt month.
- Betas are estimated through day t−1.

**Signal**

- **Residual.** Day t's residual is the stock's return minus the betas times day t's factor returns.
- **5-day z-score.** z = (sum of the last 5 daily residuals) ÷ (σ_ε × √5), with σ_ε from an EWMA using a 60-day half-life.
- **Score.** The negative of z.

**Entry**

- At the next open, go long the decile with the most negative z and short the decile with the most positive z.
- A name is traded only if |z| ≥ 1.
- **Skip rules:**
  - Names with earnings in the window from 2 days before to 1 day after the signal (information, not noise).
  - Names with a one-day move above 25% (M&A or news jumps).

**Exit**

- Each sleeve is unwound after 5 trading days.
- Stop-out: exit early if the residual z moves beyond 4 in the same direction, meaning the news is still arriving.

### Rebalance frequency

Daily: one sleeve per day. Signals use the close of day t and trades fill at the open of day t+1.

### Transaction cost assumptions

- The shared cost model applies (see "Shared cost model").
- For eligible names, the half-spread is 1.5–6 bps per side by liquidity tier.
- Square-root impact with Y = 0.5.
- Borrow at 0.3%/yr for general collateral and 2%/yr for small caps.
- Hedge trades: XLE at 1 bp, USL at 8 bps, UNL at 15 bps.
- **Turnover is the defining risk.** The whole book turns every 5 days, which is about 50× gross per year. At roughly 4–6 bps all-in per side, cost drag is about 4–6% a year of gross exposure, before any alpha.

### Why the edge might exist

**Mechanism**

The other side is liquidity demand that is not driven by information:
- creation and redemption baskets for XLE, XOP, OIH, VDE and AMLP;
- index rebalances;
- stop-loss and margin selling in a high-volatility sector;
- retail momentum chasing.

These traders pay for immediacy. The partial snap-back is the liquidity provider's fee. Price pressure from ETF ownership adds non-fundamental volatility to constituents that later reverts (Ben-David, Franzoni & Moussawi 2018).

**Why energy specifically**

A handful of observable factors (the oil strip, the gas strip, the sector) explain most of an energy stock's daily return. The leftover residual is therefore a cleaner read of non-fundamental pressure than in most sectors. Short-term reversal is concentrated in exactly this residual component (Da, Liu & Schaumburg 2014). Trading residuals against ETF factors is the Avellaneda & Lee (2010) design.

**Why it persists**

- Liquidity provision is a real service, and its returns rise when volatility spikes and balance sheets shrink (Nagel 2012).
- Energy volatility is high and episodic.

**What kills it**

- **Crowding by market makers and stat-arb desks.** For example, the August 2007 unwind (Khandani & Lo 2007).
- **Residuals that are really news.** Reserve revisions, well results, outages and M&A do not revert.
- **Execution costs.** They can consume the entire gross edge at this turnover.
- **Borrow squeezes** on crowded shorts.

---

## S2 — Capital-Discipline Value

The strategy ranks energy companies within their sub-industry on free cash flow, payouts, reinvestment restraint, leverage and valuation. It buys the disciplined and shorts the growth-at-any-cost names.

### Universe and position sizing

**Eligible securities**

- Energy equities tagged `core` or `borderline`, ADRs included.
- Excluded:
  - royalty trusts, which pass cash through and make no reinvestment decisions;
  - uranium names, which are mostly pre-production so free-cash-flow metrics are meaningless;
  - gas-fired power solutions.

**Screens**

- Market cap ≥ $1B.
- Price ≥ $5.
- 60-day median daily dollar volume ≥ $5M.
- At least 8 quarters of financial statements.

Today 157 names pass.

**Ranking buckets**

| Bucket | Sub-industries |
|---|---|
| Upstream | Integrated, E&P, minerals, coal |
| Services | Oilfield services, drilling |
| Downstream | Refining & marketing, alternative fuels |
| Midstream | Midstream, LNG |
| Shipping | Tankers & gas carriers |

Buckets with fewer than 6 eligible names merge into upstream.

**Position sizing**

- **Selection.** Long the top 20% and short the bottom 20% within each bucket.
- **Bucket weights.** Each bucket's gross is proportional to its share of eligible market cap, and each bucket is dollar-neutral.
- **Within a leg.** Names are equal-weighted, capped at 5% of the leg.
- **Book size.** The long book is 50% of NAV and the short book 50%.
- **Shorts.** MLP units are excluded from shorts because of borrow and K-1 frictions.
- **Variant.** A long-only version, holding the top 40% against VDE, is reported as an implementable variant.

### Entry/exit logic and factors

**Factors and weights**

All factors are trailing twelve months and point-in-time: a statement must have been filed at least 2 trading days before the rebalance. Each is winsorized at the 2.5th/97.5th percentiles within its bucket and then z-scored.

| Factor | Definition | Weight |
|---|---|---:|
| Free-cash-flow yield | (CFO − capex) ÷ enterprise value | +0.30 |
| Shareholder yield | (dividends + buybacks − share issuance) ÷ market cap | +0.20 |
| Reinvestment rate | Capex ÷ CFO | −0.15 |
| Asset growth | Total assets, year over year | −0.15 |
| Leverage | Net debt ÷ EBITDA, capped at 5× | −0.10 |
| Valuation | EV ÷ EBITDA | −0.10 |

**Entry and exit**

- **Entry** into the top or bottom quintile at the quarterly rebalance.
- **Buffer.** A held long stays until it drops out of the top 30%; shorts mirror this with the bottom 30%.
- **Corporate actions.**
  - Acquired names are held to deal close, with the terms-based delisting return.
  - Bankruptcies take their delisting return.

### Rebalance frequency

Quarterly, on the 15th trading day of February, May, August and November, after most 10-Qs and 10-Ks are filed. There is no trading between rebalances except for corporate actions.

### Transaction cost assumptions

- The shared tiered spread and square-root impact apply.
- Borrow is 0.3%/yr for large and mid caps and 2%/yr for small caps.
- The buffer keeps turnover around 100–200% a year per leg.
- The expected cost drag is well under 1% a year, and capacity is far above S1.

### Why the edge might exist

**Mechanism**

- The other side is investors who extrapolate commodity prices and production growth, and managements that expand at cycle peaks.
- Growth capex in a commodity business competes returns down toward the marginal cost of supply.
- Markets are slow to penalize the value destruction. This is the asset-growth and investment anomaly (Cooper, Gulen & Schill 2008; Titman, Wei & Xie 2004; the investment and profitability factors of Fama & French 2015).
- Total payouts predict returns (Boudoukh, Michaely, Richardson & Roberts 2007). Cash returned to shareholders cannot be misallocated.
- Ranking within the sub-industry strips out the commodity-cycle bet (Asness, Porter & Stevens 2000).

**Why energy specifically**

- It is the most capital-intensive, boom-bust sector, where the gap between reinvestment and payout is widest.
- ESG exclusions and divestment raise its cost of capital (Bolton & Kacperczyk 2021). Firms that self-fund and pay out collect that premium without dilution.

**What kills it**

- **Value traps.** Low reinvestment can mean reserves are being depleted. The spec has no reserve-replacement metric.
- **Regime shifts.** Early upcycles can reward growth.
- **Crowding.** The "capital discipline" theme has been crowded since 2021.
- **Thin buckets** make rankings noisy.
- **Data quality.** Point-in-time errors in the fundamentals can do damage.

---

## S3 — Energy Trend & Carry

The strategy trades trends in the energy complex's liquid ETPs, and tilts the commodity funds toward whichever side of the futures curve pays roll yield.

### Universe and position sizing

**Instruments**

- Equity ETFs: XLE, XOP, OIH, AMLP, URA.
- Commodity futures funds: USO (WTI), UNG (natural gas), UGA (gasoline).
- USL and UNL, the 12-month strip funds, serve only as carry proxies.
- Idle cash sits in BIL.
- All history must begin at least 2011, which these instruments satisfy.

**Position sizing**

- **Raw weight.** The signal ÷ the instrument's EWMA volatility (60-day half-life).
- **Portfolio scaling.** Scaled so the portfolio's ex-ante volatility, using the EWMA covariance, equals 12% a year.
- **Caps.** 50% of NAV per instrument and 200% of NAV gross.
- **Shorts** are allowed. A long-only-with-cash variant is reported alongside.

### Entry/exit logic and factors

**Trend (all 8 instruments)**

The mean of sign(excess return vs T-bills) over 21, 63, 126 and 252 trading days, giving a value in [−1, 1].

**Carry (commodity funds)**

- The 63-day return of the front fund minus that of the 12-month strip fund, annualized:
  - USO − USL for crude.
  - UNG − UNL for gas.
  - UGA uses the USO − USL reading.
- Positive means backwardation, where holding futures earns roll yield.
- Mapped to −1, 0 or +1 with a ±2%/yr deadband.
- If futures-curve data (CL1–CL12, NG1–NG12) is available, use the curve slope instead and report both.

**Combined signal and trading rules**

- **Signal.** Equity ETFs use trend. Commodity funds use 0.5 × trend + 0.5 × carry.
- **Entry and exit.** Positions follow the target weights. A position closes when its signal reaches 0 or flips sign.
- **No-trade band.** Skip any change smaller than 25% of the target or 3% of NAV.
- **Risk brake (checked daily).** If 5-day realized portfolio volatility exceeds 2× target, scale all positions back to target at once.

### Rebalance frequency

Weekly. Signals use Friday's close and trades fill at Monday's open.

### Transaction cost assumptions

- **Half-spreads:** XLE 1 bp, XOP 1.5, USO 1.5, OIH 3, AMLP 3, URA 4, UNG 4, UGA 10 bps.
- **Impact** is negligible below about $10M of NAV. The square-root model applies above that.
- **Borrow:** 0.3–2%/yr.
- **Roll costs and expense ratios** are already inside ETP prices. The backtest must use traded ETP prices, not spot or futures indices, or it will overstate the commodity leg.
- **Turnover** is about 5–10× NAV a year.

### Why the edge might exist

**Mechanism**

- **Slow physical adjustment.** Energy supply takes years to respond, and inventories absorb shocks gradually, so returns are autocorrelated (Gorton, Hayashi & Rouwenhorst 2013).
- **Trend evidence.** Time-series momentum is among the most robust effects across asset classes and a century of data (Moskowitz, Ooi & Pedersen 2012; Hurst, Ooi & Pedersen 2017).
- **Carry.** Producers hedge by selling forward. Speculators who take the other side earn a premium, largest when inventories are tight and the curve is backwardated (de Roon, Nijman & Veld 2000; Erb & Harvey 2006; Szymanowska et al. 2014; Koijen, Moskowitz, Pedersen & Vrugt 2018).

**Why it persists**

- Investors under-react to slow shifts such as OPEC policy and shale capex cycles, then herd late.
- Many holders cannot short or hold futures.
- Energy crashes (2014–16, 2020) are prolonged, which lets a short trend position pay when long-only holders suffer.

**What kills it**

- **Whipsaw regimes.** Examples are range-bound oil in 2016–19 and headline spikes that reverse, such as the April 2026 ceasefire snap-back.
- **Low breadth.** Eight highly correlated instruments give little diversification.
- **ETP structural events.** USO's 2020 restructuring is an example.
- **Noise** in the ETP-based carry proxy.

---

## S4 — Commodity-Hedged Earnings Drift

The strategy buys or sells energy stocks after earnings when the firm-specific part of the news is strongly positive or negative. It hedges out the sector and commodity moves and holds for up to two months.

### Universe and position sizing

**Eligible securities**

- Energy equities tagged `core` or `borderline`.
- Common shares, partnership units and LLC units.
- ADRs and other foreign-primary listings are excluded because home-market reporting (6-K, often semiannual) makes the event calendar inconsistent.
- Royalty trusts and gas-fired power solutions are excluded.

**Screens**

- Market cap ≥ $500M.
- Price ≥ $5.
- 60-day median daily dollar volume ≥ $5M.
- At least 8 prior quarters of earnings.

Today 165 names pass the static and snapshot screens, or roughly 600–650 events a year.

**Position sizing**

- **Base weight.** 3% of NAV per event.
- **Volatility scaling.** Multiplied by (median residual volatility ÷ the stock's own) and bounded to 1–5% of NAV.
- **Position limits.** At most 20 longs and 20 shorts at once. When full, take the highest |score|.
- **Hedging.** Each position is hedged with its sub-industry ETF, USL and UNL at estimated betas, re-sized weekly. The hedge ETF comes from the `hedge_etf` mapping in `strategies.toml`.

### Entry/exit logic and factors

**Event timing.** Day 0 is the first session that can react to the release. That is the same day for a pre-market release and the next day for an after-close release.

**EAR (earnings announcement return)**

- The cumulative abnormal return from day −2 to day +1.
- Abnormal means relative to the hedge ETF, USL and UNL, with betas estimated from day −252 to day −10.
- Standardized by residual volatility.
- This removes the part of the announcement move that the commodity tape explains.

**SUE (standardized unexpected earnings)**

- (Reported EPS − consensus EPS) ÷ the standard deviation of the past 8 surprises.
- If consensus is missing, a seasonal random walk is used instead: (EPS this quarter − EPS four quarters ago) ÷ σ.

**Score and trading rules**

- **Score.** 0.5 × EAR percentile + 0.5 × SUE percentile. Percentiles are measured against the trailing 12 months of events only, so there is no look-ahead.
- **Long entry** at the open of day +2 if the score is at or above the 80th percentile and both EAR and SUE are positive.
- **Short entry** at the same time if the score is at or below the 20th percentile and both are negative.
- **Exit** after 40 trading days, or 3 trading days before the next scheduled earnings, whichever comes first.

### Rebalance frequency

Event-driven: entries and exits are checked daily, and hedges are re-sized weekly.

### Transaction cost assumptions

- The shared model applies: tiered half-spreads and square-root impact.
- Small-cap entries carry an extra 5 bps for post-event spread widening.
- Hedge ETFs cost 1–15 bps.
- Borrow at 0.3–2%/yr. Hard-to-borrow names are excluded from shorts.
- Each position slot turns over about 6× a year.

### Why the edge might exist

**Mechanism: anchoring and attention**

- Energy investors model earnings as commodity price × volume, and analysts anchor on the commodity deck and company guidance.
- Firm-specific execution gets under-weighted. That includes well productivity, cost deflation, midstream contract wins and the pace of capital returns.
- The commodity-hedged EAR isolates exactly that component.
- Energy reporting is compressed into about three weeks, with peers reporting on the same days. That is the setting where limited attention produces drift (Hirshleifer, Lim & Teoh 2009; DellaVigna & Pollet 2009).
- Heavy passive ownership leaves less price-sensitive capital to correct the gap.

**Evidence base**

- Post-earnings drift (Bernard & Thomas 1989, 1990).
- Earnings momentum alongside price momentum (Chan, Jegadeesh & Lakonishok 1996).
- Drift after announcement returns (Brandt, Kishore, Santa-Clara & Venkatachalam 2008).

**What kills it**

- **Decay in large caps.** Drift has shrunk for large caps (Martineau 2022) and survives mainly in less liquid names, where costs bite (Chordia et al. 2009).
- **Consensus data** is patchy and survivorship-biased for delisted firms.
- **Macro shocks** can swamp idiosyncratic drift during 40-day holds.

---

## Shared cost model

Liquidity tiers use the 60-day median daily dollar volume (ADV) at trade time. All figures are per side.

| Tier | ADV | Half-spread |
|---|---|---:|
| A | ≥ $200M | 1.5 bps |
| B | ≥ $50M | 3 bps |
| C | ≥ $20M | 6 bps |
| D | ≥ $5M | 12 bps |
| — | < $5M | Not tradable |

**Other components**

- **Market impact:** 10⁴ × Y × σ_daily × √(order ÷ ADV) bps, with Y = 0.5 as base and 1.0 as stress. This is the square-root law (Almgren et al. 2005; Tóth et al. 2011).
- **Fees:**
  - $0 commission (Alpaca) plus a 0.5 bp allowance for fees and rebates.
  - The SEC Section 31 fee on sells, at 0.28 bps. Update it to the current rate.
- **Borrow:**
  - 0.3%/yr for general collateral.
  - 2%/yr for small caps and MLPs.
  - Names above 5%/yr are excluded from shorts.
- **ETPs:** expense ratios and futures roll are already inside prices, so no separate charge.
- **Reporting:** every result is shown at 1× and 2× costs, and at $1M, $10M and $100M of capital.

## Shared backtest requirements

These make the four results comparable.

**Setup**

- **Period:** 2012-01-03 to 2026-09-29, reported whole and split into 2012–2019 and 2020–2026.
  - The split covers the 2014–16 crash, 2020, the 2021–22 boom and the 2026 war shock.
- **Timing:** signals at the close of day t, fills at the open of day t+1. No same-bar fills.

**Universe and prices**

- **Universe:** point-in-time, including dead names with their delisting returns. The 2026-09-30 universe must not be used historically.
- **Why it matters for energy:** it has unusually heavy survivorship.
  - Bankruptcies include Chesapeake, Whiting, Oasis, California Resources, Denbury, Diamond Offshore, Valaris and Noble in 2020, and Weatherford in 2019.
  - Acquisitions include Pioneer and Marathon Oil in 2024, Hess and ChampionX in 2025, and Civitas, Coterra and Helix in 2026.
- **Prices:** total-return (split- and dividend-adjusted) for stocks and MLPs, including distributions. Traded prices for ETPs.

**Parameters and variants**

- The parameters in `strategies.toml` are the pre-registered spec. The listed variants are reported next to the base case, never swapped in after the fact.

**Report for each strategy**

- **Returns and risk:**
  - CAGR, volatility, Sharpe and Sortino.
  - Maximum drawdown and its duration.
- **Trading:**
  - Turnover and cost drag.
  - Hit rate and average holding period.
- **Exposures:** beta and correlation to XLE, SPY and USO.
- **Capacity curve.**
- **Regime splits:** oil up vs down quarters, and high vs low VIX.
- **Correlations:** the pairwise correlation matrix of the four strategies.

## Data the backtest needs

| Need | Used by | Candidate source |
|---|---|---|
| Daily adjusted OHLCV, 2011→, about 200 stocks and 13 ETPs | All | Alpha Vantage `TIME_SERIES_DAILY_ADJUSTED` (premium) or another vendor |
| Point-in-time membership, including delisted names | S1, S2, S4 | Alpha Vantage `LISTING_STATUS` (`date=`, `state=delisted`) plus the sub-industry mapping |
| Quarterly statements with filing dates | S2 | SEC XBRL company facts (sec.gov); Alpha Vantage fundamentals as a fallback (not point-in-time) |
| Earnings dates, timing and consensus EPS | S4 | Alpha Vantage `EARNINGS`; 8-K Item 2.02 dates as a fallback |
| Futures curves (optional, for S3 carry) | S3 | Exchange settlement data; the USO−USL and UNG−UNL proxies otherwise |
| T-bill returns | S3, cash | BIL |

**Environment constraint (as of 2026-09-30)**

- The free Alpha Vantage key allows 25 core calls a day. The analytics endpoint is limited only by bursts.
- This session's network policy blocks sec.gov, Yahoo Finance, Stooq and Alpaca data.
- The backtest therefore needs either a premium data key or those hosts allowed.

## References

- Almgren, Thum, Hauptmann & Li (2005), "Direct Estimation of Equity Market Impact", *Risk*.
- Asness, Porter & Stevens (2000), "Predicting Stock Returns Using Industry-Relative Firm Characteristics", working paper.
- Avellaneda & Lee (2010), "Statistical Arbitrage in the US Equities Market", *Quantitative Finance*.
- Ben-David, Franzoni & Moussawi (2018), "Do ETFs Increase Volatility?", *Journal of Finance*.
- Bernard & Thomas (1989), "Post-Earnings-Announcement Drift: Delayed Price Response or Risk Premium?", *JAR*.
- Bernard & Thomas (1990), *JAE*.
- Bolton & Kacperczyk (2021), "Do Investors Care about Carbon Risk?", *JFE*.
- Boudoukh, Michaely, Richardson & Roberts (2007), "On the Importance of Measuring Payout Yield", *Journal of Finance*.
- Brandt, Kishore, Santa-Clara & Venkatachalam (2008), "Earnings Announcements are Full of Surprises", working paper.
- Chan, Jegadeesh & Lakonishok (1996), "Momentum Strategies", *Journal of Finance*.
- Chordia, Goyal, Sadka, Sadka & Shivakumar (2009), "Liquidity and the Post-Earnings-Announcement Drift", *FAJ*.
- Cooper, Gulen & Schill (2008), "Asset Growth and the Cross-Section of Stock Returns", *Journal of Finance*.
- Da, Liu & Schaumburg (2014), "A Closer Look at the Short-Term Return Reversal", *Management Science*.
- de Roon, Nijman & Veld (2000), "Hedging Pressure Effects in Futures Markets", *Journal of Finance*.
- DellaVigna & Pollet (2009), "Investor Inattention and Friday Earnings Announcements", *Journal of Finance*.
- Erb & Harvey (2006), "The Strategic and Tactical Value of Commodity Futures", *FAJ*.
- Fama & French (2015), "A Five-Factor Asset Pricing Model", *JFE*.
- Gorton, Hayashi & Rouwenhorst (2013), "The Fundamentals of Commodity Futures Returns", *Review of Finance*.
- Hirshleifer, Lim & Teoh (2009), "Driven to Distraction", *Journal of Finance*.
- Hurst, Ooi & Pedersen (2017), "A Century of Evidence on Trend-Following Investing", *JPM*.
- Jegadeesh (1990), "Evidence of Predictable Behavior of Security Returns", *Journal of Finance*.
- Khandani & Lo (2007), "What Happened to the Quants in August 2007?", working paper.
- Koijen, Moskowitz, Pedersen & Vrugt (2018), "Carry", *JFE*.
- Lehmann (1990), "Fads, Martingales, and Market Efficiency", *QJE*.
- Martineau (2022), "Rest in Peace Post-Earnings Announcement Drift", *Critical Finance Review*.
- Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum", *JFE*.
- Nagel (2012), "Evaporating Liquidity", *RFS*.
- Szymanowska, de Roon, Nijman & van den Goorbergh (2014), "An Anatomy of Commodity Futures Risk Premia", *Journal of Finance*.
- Titman, Wei & Xie (2004), "Capital Investments and Stock Returns", *JFQA*.
- Tóth et al. (2011), "Anomalous Price Impact and the Critical Nature of Liquidity in Financial Markets", *PRX*.
