"""
FinViz → Alpaca technical scan CLI (writes saved_data/scan_results.csv).

Run from repo root:
  poetry run python -m Streaming_Method.scanner
"""

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from stock_picker.stock_screener import get_tickers

from Streaming_Method.streaming_alpaca import StreamingAlpacaConnection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    logging.info("Starting Stock Scanner (Streaming_Method)")

    conn = StreamingAlpacaConnection(paper=True)

    account = conn.get_account()
    logging.info("Account Cash: $%s", account.cash)

    logging.info("Fetching tickers from FinViz...")
    tickers = get_tickers()

    if not tickers:
        logging.error("No tickers found from screener")
        return

    logging.info("Found %s tickers to analyze", len(tickers))

    results_df = conn.scan_stocks(tickers, days=90)

    if "error" in results_df.columns:
        results_df = results_df[results_df["error"].isna()]

    output_df = pd.DataFrame(
        {
            "Ticker": results_df["ticker"],
            "Latest Price": results_df["price"],
            "Engulfing Signal": results_df["engulfing_latest"].fillna("Neutral"),
            "Momentum Trend": results_df.apply(
                lambda row: (
                    "Bullish"
                    if row["bullish_momentum"]
                    else ("Bearish" if row["bearish_momentum"] else "Neutral")
                ),
                axis=1,
            ),
            "Hammer Signal": results_df["hammer_recent"].apply(
                lambda x: f"{int(x)} recent" if x > 0 else ""
            ),
            "Doji Signal": results_df["doji_type"].fillna("Neutral"),
        }
    )

    out_path = _ROOT / "saved_data" / "scan_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_path, index=False)
    logging.info("Results saved to %s", out_path)

    logging.info("Stock Scanner completed")


if __name__ == "__main__":
    main()
