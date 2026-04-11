"""
Alpaca streaming + multi-ticker technical scan (pattern / scanner path).
"""

import logging
import os

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from alpaca_trade_api.stream import Stream
import pandas as pd
from dotenv import load_dotenv

from alpaca_trading import AlpacaConnection
from techAnalysis import TechnicalAnalysis

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

logger = logging.getLogger(__name__)


class StreamingAlpacaConnection(AlpacaConnection):
    """
    Extends REST client with websocket streaming and TechnicalAnalysis
    for hammer-on-bar trades and batch scanning.
    """

    def __init__(self, paper=True):
        super().__init__(paper=paper)

        stream_url = (
            "https://paper-api.alpaca.markets"
            if paper
            else "https://api.alpaca.markets"
        )
        self.stream = Stream(
            API_KEY,
            SECRET_KEY,
            base_url=stream_url,
            data_feed="iex",
        )
        self.ta = TechnicalAnalysis()

    async def on_bar(self, bar):
        """Handle a new bar; place a small bracket order on hammer detection."""
        print(f"New bar: {bar.symbol} Close: {bar.close}")

        bar_data = {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }

        if self.ta.hammerDetect(bar_data):
            print("Hammer Detected, Placing order.")
            order = self.place_bracket_order(
                bar.symbol, 1, bar.close, bar.close + 0.4, bar.close - 0.4
            )
            print(f"Order: {order}")

    def start_streaming(self, symbol="SPY"):
        """Subscribe to real-time bars (blocks until stopped)."""
        self.stream.subscribe_bars(self.on_bar, symbol)
        print(f"Starting stream for {symbol}...")
        self.stream.run()

    def scan_stocks(self, tickers, days=90):
        """
        Scan tickers with TechnicalAnalysis.analyze_stock and return a DataFrame.
        """
        results = []

        logger.info("Scanning %s tickers with %s days of data", len(tickers), days)

        for i, ticker in enumerate(tickers):
            try:
                logger.info("Analyzing %s (%s/%s)", ticker, i + 1, len(tickers))

                df = self.get_historical_data(
                    ticker, days=days, timeframe=TimeFrame.Day
                )

                if df.empty:
                    logger.warning("No data for %s", ticker)
                    continue

                analysis = self.ta.analyze_stock(df, ticker)
                results.append(analysis)

            except Exception as e:
                logger.error("Error analyzing %s: %s", ticker, e)
                results.append({"ticker": ticker, "error": str(e)})

        return pd.DataFrame(results)
