"""
Alpaca REST client for ML trading and data collection.

Streaming, FinViz scanning, and pattern-based bar logic live under Streaming_Method/.
"""

import logging
import os
from datetime import datetime, timedelta

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")


class AlpacaConnection:
    """REST-only Alpaca client (historical data, account, bracket orders)."""

    def __init__(self, paper=True):
        self.paper = paper
        base_url = (
            "https://paper-api.alpaca.markets"
            if paper
            else "https://api.alpaca.markets"
        )

        self.api = tradeapi.REST(
            key_id=API_KEY,
            secret_key=SECRET_KEY,
            base_url=base_url,
        )

    def get_account(self):
        return self.api.get_account()

    def get_historical_data(self, symbol, days=1, timeframe=TimeFrame.Minute):
        """Get historical bars."""
        end = datetime.now()
        start = end - timedelta(days=days)

        bars = self.api.get_bars(
            symbol,
            timeframe,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            feed="iex",
        ).df

        return bars

    def place_bracket_order(self, symbol, qty, entry_price, take_profit, stop_loss):
        """Submit a buy bracket order (limit entry + take-profit + stop-loss)."""
        return self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="limit",
            limit_price=entry_price,
            time_in_force="gtc",
            order_class="bracket",
            take_profit={"limit_price": take_profit},
            stop_loss={"stop_price": stop_loss},
        )
