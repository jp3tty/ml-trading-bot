"""
Alpaca REST client for ML trading and data collection.

Streaming, FinViz scanning, and pattern-based bar logic live under Streaming_Method/.
"""

import logging
import os
import re
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

# Alpaca lists ETFs/ETNs under asset_class="us_equity", so the most-actives
# screener surfaces them alongside common stock. The BUY/SELL models were
# trained only on individual-company price action, where a -2% stop means
# something; a 3x leveraged or inverse fund (SOXL, SQQQ, TSLL, …) or a crypto
# trust (IBIT, BITO) has structurally different volatility and blows through
# that stop on noise.
#
# Matched on asset name. "ETF"/"ETN" catches most; ProShares labels its
# leveraged products "UltraPro / UltraShort" with no "ETF" token, so fund
# issuers and leverage phrases ("2x", "Bull 3", "Daily Target") are matched
# too. Swept against Alpaca's full active-equity list: catches every ProShares
# /Direxion/GraniteShares leveraged & inverse fund with zero operating-company
# false positives. Every drop is logged so a future false positive is visible.
_FUND_NAME_RE = re.compile(
    r"\b(ETF|ETN)\b"
    r"|ProShares|Direxion|GraniteShares|Leverage Shares|\bTradr\b|Defiance"
    r"|iShares|Invesco QQQ|\bSPDR\b|VanEck|Roundhill|Themes ETF"
    r"|Ultra ?Pro|Ultra ?Short"
    r"|\b\d(?:\.\d)?x\b|\bBull \d|\bBear \d|Daily Target",
    re.IGNORECASE,
)


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

    def get_most_active_symbols(self, top=50, min_price=10.0,
                                min_dollar_volume=100_000_000,
                                exclude_funds=True):
        """Return the most-active US equities from Alpaca's own screener API.

        Replaces the FinViz HTML scraper (2026-07-27) — FinViz was found to
        be serving deliberately corrupted ticker data to non-browser requests
        (an anti-scraping defense), which silently zeroed out the BUY
        watchlist for ~10 days. This endpoint isn't wrapped by the legacy
        alpaca-trade-api SDK, so it's called directly; it uses the same
        first-party credentials as everything else.

        Liquidity floor (2026-09-01): the raw most-actives list is ranked by
        *share* volume, so it fills up with sub-$5 penny names (RITR $0.09,
        HKPD $0.20, LIDR $1.52, GPRO $1.20, …) that carry huge spreads and
        overnight-gap risk — exactly the tickers whose stop-losses gapped
        through to -11%…-18% fills during the FinViz era. We now require:

          * price >= `min_price`  (raised $5 -> $10)
          * price * consolidated daily volume >= `min_dollar_volume`

        Also filters out share classes/warrants/units (symbols containing '.').
        The screener has no market-cap field; dollar-volume is the liquidity
        proxy that matters for fill quality anyway.

        Fund filter (2026-09-01): with `exclude_funds`, drops ETFs/ETNs by
        asset name — leveraged/inverse funds (SOXL, SQQQ, TSLL, …), bond ETFs
        (SGOV, HYG, LQD) and crypto trusts (IBIT, BITO) don't behave like the
        individual equities the models were trained on and routinely trip a
        -2% stop on noise.
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

        rows = {
            row["symbol"]: row
            for row in resp.json().get("most_actives", [])
            if row.get("symbol") and "." not in row["symbol"]
        }
        candidates = list(rows)
        if not candidates:
            logging.warning("most-actives screener returned no usable symbols")
            return []

        funds = set()
        if exclude_funds:
            try:
                assets = {
                    a.symbol: a for a in
                    self.api.list_assets(status="active", asset_class="us_equity")
                }
                for s in candidates:
                    a = assets.get(s)
                    if a is not None and _FUND_NAME_RE.search(a.name or ""):
                        funds.add(s)
            except Exception as e:
                logging.warning(
                    f"could not fetch asset names for fund filter ({e}) — "
                    f"skipping ETF exclusion this run"
                )

        snapshots = self.api.get_snapshots(candidates, feed="iex")
        symbols = []
        dropped_price, dropped_liquidity, dropped_noprice = [], [], []
        dropped_fund = []
        for symbol in candidates:
            if symbol in funds:
                dropped_fund.append(symbol)
                continue
            snap = snapshots.get(symbol)
            price = None
            if snap:
                if snap.latest_trade and snap.latest_trade.price:
                    price = snap.latest_trade.price
                elif snap.daily_bar and snap.daily_bar.close:
                    price = snap.daily_bar.close
            if not price:
                dropped_noprice.append(symbol)
                continue
            if price < min_price:
                dropped_price.append(symbol)
                continue
            dollar_volume = price * rows[symbol].get("volume", 0)
            if dollar_volume < min_dollar_volume:
                dropped_liquidity.append(f"{symbol}(${dollar_volume/1e6:.0f}M)")
                continue
            symbols.append(symbol)

        logging.info(
            f"most-actives: {len(symbols)}/{len(candidates)} passed filters "
            f"(min_price=${min_price:.0f}, min_$vol=${min_dollar_volume/1e6:.0f}M, "
            f"exclude_funds={exclude_funds})"
        )
        if dropped_fund:
            logging.info(f"  dropped — ETF/ETN: {dropped_fund}")
        if dropped_price:
            logging.info(f"  dropped — under ${min_price:.0f}: {dropped_price}")
        if dropped_liquidity:
            logging.info(f"  dropped — thin $-volume: {dropped_liquidity}")
        if dropped_noprice:
            logging.warning(f"  dropped — no price data: {dropped_noprice}")
        if len(symbols) < 5:
            logging.warning(
                f"most-actives watchlist unusually small ({len(symbols)}) — "
                f"liquidity floor may be too tight or the market is thin today"
            )
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
