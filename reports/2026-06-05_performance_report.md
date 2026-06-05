# ML Trading Bot — Performance Report
**Period:** May 12 – June 1, 2026 (3 weeks of live paper trading)
**Generated:** 2026-06-05

---

## Strategy Overview

The bot uses two Random Forest / XGBoost ML models trained on 4-hour OHLCV bars. It runs twice daily (9:30 AM and 1:30 PM ET) via GitHub Actions. Each run:

1. **SELL pass** — checks all held positions for an ML sell signal (XGBoost, 40.1% precision / 100% recall, threshold ≈ 4%)
2. **BUY pass** — scores FinViz momentum-screened candidates (small-cap+, rel-vol >2×, 5-day positive), buys the top-5 by probability (Random Forest, 37.1% precision / 100% recall, threshold ≈ 0.5%)
3. **Risk management** — ATR-based stop loss (2×ATR), no take-profit ceiling, max 20 positions, 2% of buying power per trade (max $1,000)

---

## Closed Trade Performance (80 completed round-trips)

| Metric | Value |
|---|---|
| Total closed trades | 80 |
| Win rate | **40.0%** |
| Total net P&L | **-$401.93** |
| Avg winning trade | +$29.03 (+4.80%) |
| Avg losing trade | -$27.73 (-2.93%) |
| Profit factor | **0.70** *(needs >1.0 to be net-positive)* |

### Top 5 Winners
| Symbol | Shares | Entry | Exit | P&L |
|---|---|---|---|---|
| NL | 123 | $6.07 | $7.91 | **+$226** (+30.3%) |
| NL | 79 | $6.46 | $8.48 | **+$159** (+31.3%) |
| NL | 41 | $6.07 | $8.48 | **+$99** (+39.7%) |
| KRNT | 53 | $17.75 | $18.41 | +$35 (+3.7%) |
| KELYA | 96 | $10.31 | $10.64 | +$32 (+3.3%) |

### Top 5 Losers
| Symbol | Shares | Entry | Exit | P&L |
|---|---|---|---|---|
| LOCO | 58 | $17.14 | $14.76 | **-$138** (-13.9%) |
| PHOE | 50 | $19.62 | $17.02 | **-$130** (-13.3%) |
| CIB | 11 | $84.44 | $73.63 | **-$119** (-12.8%) |
| SKM | 19 | $50.28 | $44.37 | **-$112** (-11.8%) |
| BMRN | 16 | $60.11 | $55.30 | **-$77** (-8.0%) |

**Open Positions:** 23 symbols, ~$26,409 cost basis still at risk. The portfolio carries 5 separate lots of NL alone at prices from $6.46–$8.31.

---

## Critical Observations

**1. The P&L is propped up by a single outlier.** Three NL trades generated +$485 in gains — without them, the bot's net P&L on the other 77 trades is approximately **-$887**. The strategy is not working; one lucky small-cap run is masking it.

**2. A 40% win rate cannot sustain a net-positive system at near-even win/loss sizes.** With avg win (+$29) only slightly larger than avg loss (-$28), the bot needs a win rate above ~49% just to break even. At 40% it will lose money systematically. The 100% recall / ~37% precision training objective is the root cause — the model flags nearly everything as a buy to avoid missing winners, but most flags are wrong.

**3. The June 1 batch was a near-disaster.** Six positions (LOCO, CIB, SKM, TWLO, HNGE, SAIC) were bought and then sold within the same session at losses of 8–14%. The ML sell model fired on all of them within 1.5 minutes of each other with 65–69% confidence. Two things went wrong: (a) several stop-loss levels were set at exactly $0.39 below entry (suggesting an ATR-calculation failure or flat fallback), and (b) the stocks may have been traded in extended hours (~7 PM ET), where illiquid prices and wide spreads can cause sharp moves.

**4. Position concentration / pyramid buying.** The bot does not prevent buying additional lots of symbols already held in paper positions. NL has 5 open lots at different price levels from $6.46 to $8.31, MLP has 2, GDEV has 2, AAPG has 2. The Alpaca position-check logic works against live Alpaca positions — but when a symbol was exited and re-screened the next day, it gets re-entered freely.

---

## 3 Recommendations

### 1. Retrain for Precision, Not Recall — and Raise the Confidence Threshold

The current 100% recall / 37% precision training objective is fundamentally misaligned with profitable trading. Recall 100% means the model never misses a true winner — but only 1 in 3 signals is right. In live trading, transaction costs and bid-ask spread mean you need to be right more than half the time.

**Action:** Rerun `binary_search.py` optimizing for F-beta with β < 1 (e.g., β=0.5) to weight precision higher than recall. Target 55%+ precision even if recall drops to 60–65%. Simultaneously raise `min_confidence` from `0.45` to `0.58–0.62` in `ml_trader.py`. This will reduce trade frequency dramatically but should flip the win rate above 50%.

### 2. Enforce One-Lot-Per-Symbol and a Hard Max-Hold Period

Two structural bugs are bleeding capital: (a) the same screener ticker gets re-bought across multiple sessions even when you already have exposure, and (b) there is no maximum holding period — positions that the sell model ignores are held indefinitely while the account's buying power gets consumed.

**Action:** Before executing any BUY, check not just Alpaca's live positions but also `orders.csv` — if a symbol has any open BUY without a matching SELL, skip it entirely. Add a forced close rule in the SELL pass: any position held more than 7 trading days without a sell signal gets closed regardless. This creates a backstop that prevents indefinite capital lockup and eliminates the NL-style pyramid buildup.

### 3. Diagnose and Fix the ATR Stop-Loss Calculation for Extended-Hours Trades

The June 1 second batch shows stops set to exactly entry − $0.39 for all five positions (TWLO, HNGE, CIB, SAIC, SKM), regardless of price level. A $239 stock and a $50 stock should not have the same $0.39 stop — this indicates the ATR calculation returned `None` and the fallback also failed. These stops were either too tight (immediate stop-outs) or non-functional (stocks fell well below stop levels).

**Action:** Add a guard in `_calculate_atr()` — if the computed stop would be less than 1% below entry, log a warning and use a 3% fallback. Add a time-of-day check: do not execute new BUY orders after 3:45 PM ET. Several June 1 buys were placed at ~7 PM ET, deep in after-hours trading where spreads widen and ML features trained on regular-hours 4h bars become unreliable.

---

## Bottom Line

The bot's core mechanics are sound — the dual-model pipeline, ATR stops, position-limit cap, and screener integration are all reasonable architecture. But the models are optimized for the wrong objective, and two execution bugs (pyramid buying + after-hours entries) are amplifying losses. The NL windfall in week one makes the results look survivable — but it was a single outlier that would not repeat reliably. Addressing the precision/recall tradeoff in the model is the highest-leverage fix before considering live capital.
