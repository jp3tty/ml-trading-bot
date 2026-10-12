# Auto Trade - ML Stock Trading System

Automated stock trading system using separate binary ML models for entry and exit signal detection. Pulls a filtered most-actives watchlist from Alpaca's screener, scores it with independent BUY and SELL detectors trained on 4-hour candlestick data, and executes GTC bracket orders through Alpaca's paper or live trading API. Runs twice each weekday via GitHub Actions.

## Strategy

**Early-exit short-term trading** using two independent binary classifiers:

| Detector | Objective | Tuned For |
|----------|-----------|-----------|
| **BUY** | Generate high-quality entry signals | F-beta (β=0.5) — precision-weighted; target 55%+ live win rate |
| **SELL** | Detect early signs of decline | Recall — exit fast, miss fewer turns |

The system defaults to **HOLD**. A position is entered when the BUY detector fires. It exits by whichever comes first: the bracket take-profit (+2.5%) or stop-loss (−2.0%), a gated ML SELL signal (only on positions already up ≥ +0.5%), or a max-hold force-close (flat/losing positions past the BUY model's 12-bar horizon).

> **Where things stand (2026-09-28):** The pipeline is stable, but the models have not shown a real edge. Full grid searches found ROC-AUC ≈ 0.55–0.61 for BUY and ≈ 0.52–0.57 for SELL across every combination tried. Losses trace to entry quality (~60% of entries hit the stop). A 2026-09-01 backtest showed no alternative exit policy beats the current one. The next lever is new BUY features (Phase 7). See [Current Status](#current-status).

### Architectural Decision (2026-06-05)

After 3 weeks of paper trading (80 closed trades), the recall-first BUY objective produced a 40% live win rate and -$401.93 net P&L. The SELL detector did not compensate for weak entries — it fired prematurely on some positions at large losses while ignoring others entirely. The BUY detector is now optimized for **precision over recall** using **F-beta (β=0.5)**, which weights precision 4× more than recall. The minimum precision floor is raised to 50–55%. Trade frequency will decrease but expected win rate should exceed the 50% break-even threshold.

*Previous decision (2026-05-06, reversed):* The BUY detector was optimized for recall over precision on the assumption that the SELL detector would manage bad entries. Paper trading invalidated this assumption.

### Performance Summary (2026-06-05 — historical, pre-bracket-change)

| Metric | Value |
|--------|-------|
| Paper trade period | May 12 – June 1, 2026 |
| Closed trades | 80 |
| Win rate | 40.0% |
| Net P&L | -$401.93 |
| Profit factor | 0.70 |

Full analysis: [`reports/2026-06-05_performance_report.md`](reports/2026-06-05_performance_report.md)

### BUY Labeling: Triple-Barrier Method

BUY labels are generated using the **triple-barrier method** (Lopez de Prado), not a simple fixed-horizon return. For each candle, the next `horizon` bars are scanned using high/low prices:

- **Label 1 (BUY):** take-profit barrier (`+take_profit%`) is touched *before* the stop-loss barrier
- **Label 0 (NOT BUY):** stop-loss is hit first, both barriers hit the same candle (conservative tie-break), or neither barrier is hit within `horizon` candles (time barrier)

This produces meaningfully balanced labels and filters for setups with an explicit positive reward:risk ratio.

## Architecture

> For a full presentation-ready breakdown of the system design, see [ARCHITECTURE.md](ARCHITECTURE.md).

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA COLLECTION                             │
│  Alpaca API  ──▶  HistoricalCollector  ──▶  Parquet (per ticker)   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FEATURE ENGINEERING                           │
│  BinaryFeatureBuilder / BinarySellFeatureBuilder                    │
│  Modes: combined (primary) · catch22 · indicators                  │
└──────────────────┬──────────────────────────┬───────────────────────┘
                   │                          │
          ┌────────▼────────┐       ┌─────────▼───────┐
          │  BUY Detector   │       │  SELL Detector  │
          │  binary_search  │       │  sell_search    │
          │  XGBoost / RF   │       │  XGBoost / RF   │
          └────────┬────────┘       └─────────┬───────┘
                   │                          │
                   └────────────┬─────────────┘
                                │
               ┌────────────────▼────────────────┐
               │   paper_trade_validator.py (CI) │
               │   / ml_trader.py                │
               │   Pass 1: SELL all held tickers │
               │   Pass 2: BUY watchlist tickers │
               └────────────────┬────────────────┘
                                │
                       Alpaca bracket orders (SL −2.0%, TP +2.5%, GTC)
```

## Project Structure

```
ml-trading-bot/
├── alpaca_trading.py                    # Alpaca REST client + most-actives watchlist screener
├── ml_trader.py                         # Main trading orchestrator (cron job)
├── paper_trade_validator.py             # Scheduled paper-trading entry point (run by CI) + signal log
├── techAnalysis.py                      # RSI, momentum, candlestick patterns
├── dashboard/
│   ├── app.py                           # Streamlit monitoring dashboard
│   └── requirements.txt                 # Streamlit Cloud dependencies
├── .streamlit/
│   └── secrets.toml.example             # Credential setup template
├── data_collection/
│   ├── historical_collector.py          # Bulk daily data fetching
│   └── historical_collector_4h.py       # 4-hour data fetching
├── ml/
│   ├── binary_feature_builder.py        # BUY features (catch22 / indicators)
│   ├── binary_sell_feature_builder.py   # SELL features (inverted labels)
│   ├── binary_search.py                 # BUY hyperparameter grid search
│   ├── binary_sell_search.py            # SELL hyperparameter grid search
│   ├── binary_predictor.py              # BUY detector — live inference
│   ├── binary_sell_predictor.py         # SELL detector — live inference
│   ├── feature_builder.py               # (legacy — 3-class)
│   ├── predictor.py                     # (legacy — 3-class)
│   └── trainer.py                       # (legacy — ROCKET)
├── models/
│   ├── binary_search_results/           # BUY champion models + search CSVs
│   └── sell_search_results/             # SELL champion models + search CSVs
├── paper_trade_log/
│   ├── signals.csv                      # All BUY/SELL signals scored per run
│   └── orders.csv                       # All orders placed with indicators
├── saved_data/
│   ├── historical/                      # Daily parquet files
│   └── historical_4h/                   # 4-hour parquet files
├── stock_picker/
│   └── stock_screener.py                # FinViz scraper (legacy — used only by Streaming_Method/)
├── Streaming_Method/                    # Legacy pattern scanner (separate subsystem)
├── .github/workflows/                   # Twice-daily paper-trading schedule
├── reports/
│   └── YYYY-MM-DD_performance_report.md # Periodic performance analyses
├── ML_TRADING_PLAN.md                   # Implementation roadmap
├── pyproject.toml
└── README.md
```

## Setup

### Prerequisites
- Python 3.12
- Alpaca account (paper or live)
- Poetry

### Installation

```bash
git clone https://github.com/jp3tty/ml-trading-bot.git
cd ml-trading-bot
pip install poetry
poetry install
```

### Environment Variables

Create a `.env` file in the project root:

```env
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```

## Usage

### 1. Collect Historical Data

```bash
# Daily bars (used for live inference)
poetry run python data_collection/historical_collector.py

# 4-hour bars (used for model training)
poetry run python data_collection/historical_collector_4h.py
```

### 2. Train the BUY Detector

```bash
# Quick search — recommended starting point (~3 hrs, representative sample)
poetry run python ml/binary_search.py --quick --max-files 200

# Quick search on all data (~6-8 hrs, use to validate champion)
poetry run python ml/binary_search.py --quick

# Full grid search (~30-40 hrs)
poetry run python ml/binary_search.py
```

Champion model saved to `models/binary_search_results/champion_binary_<timestamp>.pkl`.

### 3. Train the SELL Detector

```bash
# Quick search — use --max-files to limit data loaded per combination (recommended)
poetry run python ml/binary_sell_search.py --quick --max-files 60

# Full grid search
poetry run python ml/binary_sell_search.py --max-files 60
```

Champion model saved to `models/sell_search_results/champion_sell_<timestamp>.pkl`.

> **Note:** Without `--max-files`, the search loads all 400+ parquet files per combination
> and can take many hours. 60 files gives a representative sample and completes in ~15 min.

### 4. Validate with Paper Trading

```bash
# Dry run — scan signals without placing orders
poetry run python paper_trade_validator.py --dry-run

# Live paper trading — places real bracket orders on Alpaca paper account
poetry run python paper_trade_validator.py

# View cumulative signal log summary
poetry run python paper_trade_validator.py --report

# Scan specific symbols
poetry run python paper_trade_validator.py --dry-run --symbols AAPL TSLA NVDA

# Override SELL confidence floor (default: 0.3)
poetry run python paper_trade_validator.py --sell-confidence 0.4
```

### 5. Run the ML Trader

```bash
# Dry run
poetry run python ml_trader.py --dry-run

# Paper trading (default)
poetry run python ml_trader.py

# Specify confidence threshold
poetry run python ml_trader.py --confidence 0.45

# Live trading
poetry run python ml_trader.py --live

# Use specific model files
poetry run python ml_trader.py \
  --model models/binary_search_results/champion_binary_<timestamp>.pkl \
  --sell-model models/sell_search_results/champion_sell_<timestamp>.pkl
```

#### ML Trader Options

| Flag | Description |
|------|-------------|
| `--dry-run` | Preview trades without executing |
| `--symbols` | Specific tickers to scan for BUY signals |
| `--confidence` | Min BUY confidence threshold (default: 0.45) |
| `--model` | Path to BUY champion `.pkl` (default: latest) |
| `--sell-model` | Path to SELL champion `.pkl` (default: latest) |
| `--live` | Use live trading instead of paper |

## Trading Loop

The scheduled GitHub Actions job (`.github/workflows/`) runs `paper_trade_validator.py --confidence 0.45 --sell-confidence 0.3` at 14:00 and 17:00 UTC, Mon–Fri. It then commits `paper_trade_log/` back to the repo. `ml_trader.py` has the same two-pass logic for manual runs. Changes to trading rules must be mirrored in both files.

Each run executes two passes:

**Pass 1 — SELL** (held positions from Alpaca):
- Fetches all open positions once via `get_held_positions()`. This pass doesn't depend on the watchlist, so held tickers are never missed.
- Runs the SELL detector on every held ticker.
- **P&L gate:** a SELL signal is ignored unless `(current_price − avg_entry_price) / avg_entry_price ≥ +0.5%` (`MIN_SELL_PNL_PCT`). The ML SELL only closes small winners. The bracket stop handles downside.
- **Max hold:** once a position has been held for `horizon` bars (12 × 4h, read from the BUY champion) *and* P&L ≤ 0, it's force-closed. Winners past the horizon keep riding toward the TP. Entry time comes from replaying `orders.csv`.

**Pass 2 — BUY** (Alpaca most-actives watchlist):
- `AlpacaConnection.get_most_active_symbols()` pulls Alpaca's top-50 most-actives screener list, then drops:
  - ETFs/ETNs (leveraged/inverse funds, bond ETFs, crypto trusts, via `_FUND_NAME_RE`)
  - share classes and warrants
  - anything under **$10**
  - anything under **$100M/day** in dollar volume

  That leaves ~15–20 names. `_validate_watchlist()` logs a loud `ERROR` if ≥50% of tickers aren't real Alpaca symbols, which catches silent data-source failures.
- Skips tickers already held, plus any ticker exited within the last **7 days** (`REENTRY_COOLDOWN_DAYS`).
- Fetches **90 calendar days** of 4H bars per candidate and skips any with fewer than 100 bars.
- Scores every candidate, ranks by probability, and buys only the **top 5** that clear the confidence threshold (capped by open slots; `MAX_POSITIONS = 20`).
- Anchors the bracket to the live ask from `get_live_price()` (falls back to last trade, then prior close). Stop-loss is **−2.0%** and take-profit is **+2.5%**, `time_in_force="gtc"`. These are wider than the 0.8%/1.0% training barriers so trades survive intraday noise on 4h bars (changed 2026-06-25).
- Position size is 2% of buying power, capped at $1,000 per trade.

### Tunable constants (`ml_trader.py` / `paper_trade_validator.py`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `STOP_LOSS_PCT` | 0.020 | Bracket stop-loss |
| `TAKE_PROFIT_PCT` | 0.025 | Bracket take-profit |
| `MIN_SELL_PNL_PCT` | 0.005 | ML SELL only fires above this gain |
| `REENTRY_COOLDOWN_DAYS` | 7 | Days before a just-exited symbol can be re-bought (0 disables) |
| `MAX_POSITIONS` | 20 | Max concurrent positions |
| `WATCHLIST_INVALID_PCT` | 0.5 | Invalid-ticker fraction that triggers the watchlist alarm |

## Models

### BUY Detector (Current Champion)

| Parameter | Value |
|-----------|-------|
| Classifier | XGBoost (200 trees, max_depth=10) |
| Window size | 30 bars (4h) |
| Horizon | 12 bars (4h) ≈ 48 hours |
| Take profit (label) | 1.0% |
| Stop loss (label) | 0.8% |
| Decision threshold | 0.814 |
| Precision | 50.0% (pinned to the floor) |
| Recall | 22.1% |
| F1 | 0.307 |
| ROC-AUC / PR-AUC | 0.567 / 0.477 |
| Sample | TP=11,523 · FP=11,519 · 264k BUY-labelled samples |
| Feature mode | combined |
| Labeling | Triple-barrier |
| Training data | 1,312 symbols (4h bars) |
| Search | Full grid, 2,016 combos, 2026-07-18 (`champion_params_20260718_155350.json`) |

**Separability finding (2026-07-25):** ROC-AUC across the whole grid ranged 0.545–0.614 (mean 0.570). The ceiling among statistically robust rows was 0.611. The current indicator/catch22 feature set doesn't carry much signal for this labeling task. More grid searches over the same features aren't expected to help.

**Previous champions:** 2026-06-07 XGBoost (threshold 0.777, 50.0% precision / 49.2% recall). Before that, a recall-first Random Forest (window=21, horizon=6, 38.2% precision, 100% recall) that produced a 40% live win rate.

### SELL Detector (Current Champion)

| Parameter | Value |
|-----------|-------|
| Classifier | Random Forest |
| Window size | 20 bars (4h) |
| Horizon | 5 bars (4h) |
| Sell threshold | 0.5% |
| Decision threshold | 0.597 |
| Runtime confidence floor | **0.30** |
| Precision | 41.3% |
| Recall | 99.9% |
| F1 | 0.585 |
| Feature mode | catch22 |
| Training data | 481 symbols, 2023-01-01 – 2026-05-28 |
| Search date | 2026-05-30 |

Optimised for recall (fast exits). Champion selection requires `precision ≥ 40%`, ranked by F1.

**Status (2026-07-26):** A full 1,152-combo SELL grid search found no combination that qualified as champion (ROC-AUC 0.521–0.566), so this 2026-05-30 model is still live. In practice the +0.5% P&L gate makes it a "take small profits" trigger. The 2026-09-01 backtest put it at about +$51 net versus removing it, and every alternative exit tested (trailing stops at 1.0–2.5%, higher gates) lost money. It stays as-is.

The `sell_threshold=0.005` (0.5% decline = SELL label) was inadvertently dropped from the quick search space in earlier runs, capping achievable precision at ~34–35%. Restoring it allowed the search to find the current champion at 41.3% precision. This threshold should remain in the quick search space.

### Feature Modes

| Mode | Description | Features |
|------|-------------|----------|
| `catch22` | 22 canonical time series statistics on close + volume | 44 total |
| `indicators` | RSI, momentum, hammer, doji, engulfing, volatility | 11 × window |
| `combined` | Flattened indicators + catch22 | 11×window + 22 |

### Champion Selection

**BUY**: `precision ≥ min_precision` (searched: 0.50–0.55) → ranked by **F-beta (β=0.5)**
**SELL**: `recall ≥ 20%` and `precision ≥ 40%` → ranked by F1

The BUY threshold is chosen to maximize F-beta (β=0.5) subject to the precision floor — this weights precision 4× more than recall. The SELL threshold is chosen to maximize F1 subject to both floors.

## Output Files

| File | Description |
|------|-------------|
| `paper_trade_log/signals.csv` | Every BUY/SELL signal scored per run |
| `paper_trade_log/orders.csv` | Every order placed with entry/TP/SL/RSI/momentum |
| `models/binary_search_results/` | BUY champion `.pkl` + search results `.csv` |
| `models/sell_search_results/` | SELL champion `.pkl` + search results `.csv` |
| `saved_data/historical_4h/*.parquet` | 4-hour OHLCV data per ticker |
| `reports/` | Periodic performance analyses |

## Monitoring Dashboard

A Streamlit dashboard at `dashboard/app.py` provides live visibility into paper trading results:

| Section | Content |
|---------|---------|
| Account Summary | Portfolio value, buying power, day P&L |
| Active Positions | Open positions with entry price, current price, unrealized P&L, RSI and momentum at entry |
| Trade History | Entry/exit date, entry/exit price, P&L, exit type (Take Profit / Stop Loss / SELL Signal), entry indicators |
| Signal Log | Full history of all tickers scored per run |

After each run, the GitHub Actions workflow automatically commits updated `orders.csv` and `signals.csv` back to the repo so the dashboard always reflects the latest data.

### Run locally

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# fill in your Alpaca API keys in secrets.toml
streamlit run dashboard/app.py
```

### Deploy to Streamlit Community Cloud

1. Connect your GitHub repo at [share.streamlit.io](https://share.streamlit.io)
2. Set **Main file path** to `dashboard/app.py`
3. Add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` under **Secrets**

> **Note:** The dashboard uses the `alpaca-py` SDK (not the older `alpaca-trade-api`). Secrets only need the two keys above — no endpoint configuration required.

## Watchlist Source

The BUY watchlist comes from Alpaca's `/v1beta1/screener/stocks/most-actives` endpoint (see [Trading Loop](#trading-loop) for filters). It replaced the FinViz scraper on 2026-07-27. FinViz had started serving deliberately corrupted tickers (e.g. `OOTLY` for `OTLY`) to non-browser requests, which silently stopped all trading for ~10 days while CI still reported success. `stock_picker/stock_screener.py` still exists but is used only by `Streaming_Method/`.

> **Troubleshooting tip:** if performance looks suspiciously flat, check the date range of `signals.csv` first. It stops growing when the watchlist source breaks, even if the workflow is green.

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data collection (daily + 4h parquet) | ✅ Complete |
| 2 | Feature engineering (catch22 + indicators + combined) | ✅ Complete |
| 3 | BUY detector training + champion selection | ✅ Complete (full-grid champion 2026-07-25; weak separability confirmed) |
| 4 | BUY detector integrated into trading loop | ✅ Complete |
| 5 | SELL detector training + champion selection | ✅ Complete (full-grid confirmed no better SELL model, 2026-07-26) |
| 6 | Full BUY + SELL integration + paper trading | ✅ Running |
| 6a | Streamlit monitoring dashboard | ✅ Complete |
| 6b | Performance analysis + objective pivot | ✅ Complete (2026-06-05) |
| 6c | Watchlist hardening, re-entry cooldown, exit backtest | ✅ Complete (2026-09-01) |
| 7 | BUY model improvement — market-context features, label target, walk-forward | 🔜 Next |

**Recent history:**
- **2026-06-25:** live brackets widened to SL −2.0% / TP +2.5%.
- **2026-07-25:** full-grid BUY champion deployed; max-hold force-close added.
- **2026-07-27:** watchlist moved from FinViz to Alpaca's screener.
- **2026-09-01:** the analysis of 62 round-trips since the bracket change (−$1,264 net) found the loss was almost entirely 12 gap-through stop-outs on thin sub-$5 names. Fixes: an ETF/price/liquidity filter on the watchlist and a 7-day re-entry cooldown. A bar-level backtest showed the current exit policy beats every alternative tested.

**What's next:** Losses trace to entry quality, not exits. Phase 7 targets the BUY model directly with market-context features and a label-target review. See [ML_TRADING_PLAN.md](ML_TRADING_PLAN.md).

## Development Roadmap

See [ML_TRADING_PLAN.md](ML_TRADING_PLAN.md) for the full implementation plan and phase checklist.

## License

MIT
