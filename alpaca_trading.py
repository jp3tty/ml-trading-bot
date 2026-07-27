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
import requests
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

    def get_most_active_symbols(self, top=50, min_price=5.0):
        """Return the most-active US equities from Alpaca's own screener API.

        Replaces the FinViz HTML scraper (2026-07-27) — FinViz was found to
        be serving deliberately corrupted ticker data to non-browser requests
        (an anti-scraping defense), which silently zeroed out the BUY
        watchlist for ~10 days. This endpoint isn't wrapped by the legacy
        alpaca-trade-api SDK, so it's called directly; it uses the same
        first-party credentials as everything else.

        Filters out share classes/warrants/units (symbols containing '.')
        and anything under `min_price`, approximating the quality floor the
        old FinViz filter (market cap, relative volume, performance) enforced
        — Alpaca's screener has no equivalent fine-grained filtering.
        """
        headers = {
            "APCA-API-KEY-ID": API_KEY,
            "APCA-API-SECRET-KEY": SECRET_KEY,
        }
        resp = requests.get(
            "https://data.alpaca.markets/v1beta1/screener/stocks/most-actives",
            headers=headers,
            params={"top": top},
            timeout=15,
        )
        resp.raise_for_status()

        candidates = [
            row["symbol"] for row in resp.json().get("most_actives", [])
            if row.get("symbol") and "." not in row["symbol"]
        ]
        if not candidates:
            return []

        snapshots = self.api.get_snapshots(candidates, feed="iex")
        symbols = []
        for symbol in candidates:
            snap = snapshots.get(symbol)
            price = snap.latest_trade.price if snap and snap.latest_trade else None
            if price and price >= min_price:
                symbols.append(symbol)
        return symbols

    def get_tradable_symbols(self):
        """Return the set of active, tradable US equity symbols Alpaca knows about.

        Used to sanity-check external ticker sources (e.g. the FinViz scraper)
        against real symbols before spending a data fetch on each one.
        """
        assets = self.api.list_assets(status='active', asset_class='us_equity')
        return {a.symbol for a in assets if a.tradable}

    def get_live_price(self, symbol):
        """Return the current ask price from the latest quote, falling back to last trade price."""
        try:
            quote = self.api.get_latest_quote(symbol, feed='iex')
            for attr in ('ask_price', 'ap'):
                val = getattr(quote, attr, None)
                if val and float(val) > 0:
                    return float(val)
        except Exception:
            pass
        try:
            trade = self.api.get_latest_trade(symbol, feed='iex')
            for attr in ('price', 'p'):
                val = getattr(trade, attr, None)
                if val and float(val) > 0:
                    return float(val)
        except Exception:
            pass
        return None

    def place_bracket_order(self, symbol, qty, entry_price, stop_loss, take_profit=None):
        """Submit a buy order with stop-loss. Pass take_profit to add a TP ceiling."""
        kwargs = dict(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="limit",
            limit_price=entry_price,
            time_in_force="gtc",
            stop_loss={"stop_price": stop_loss},
        )
        if take_profit:
            kwargs["order_class"] = "bracket"
            kwargs["take_profit"] = {"limit_price": take_profit}
        else:
            kwargs["order_class"] = "oto"
        return self.api.submit_order(**kwargs)
