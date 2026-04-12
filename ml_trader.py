import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd
from alpaca_trade_api.rest import TimeFrame

from alpaca_trading import AlpacaConnection
from ml.binary_predictor import BinaryBuyPredictor
from ml.binary_sell_predictor import BinarySellPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MLTrader:
    def __init__(self, model_path=None, sell_model_path=None, paper=True):
        self.conn = AlpacaConnection(paper=paper)

        # BUY detector — loads latest champion if no path specified
        self.buy_detector = BinaryBuyPredictor(model_path=model_path)
        buy_info = self.buy_detector.get_model_info()
        logging.info(
            f"BUY detector ready | "
            f"window={buy_info['params'].get('window_size')} | "
            f"horizon={buy_info['params'].get('horizon')} | "
            f"threshold={buy_info['threshold']:.3f}"
        )

        # SELL detector — optional until a champion is trained
        self.sell_detector = None
        try:
            self.sell_detector = BinarySellPredictor(model_path=sell_model_path)
            sell_info = self.sell_detector.get_model_info()
            logging.info(
                f"SELL detector ready | "
                f"window={sell_info['params'].get('window_size')} | "
                f"horizon={sell_info['params'].get('horizon')} | "
                f"threshold={sell_info['threshold']:.3f}"
            )
        except FileNotFoundError:
            logging.warning(
                "No SELL champion found — SELL detector disabled. "
                "Run ml/binary_sell_search.py to train one."
            )

    def get_watchlist(self):
        """Symbols to scan for BUY signals — FinViz momentum screener."""
        from stock_picker.stock_screener import get_tickers
        return get_tickers()

    def get_held_positions(self):
        """Return all currently held positions as {symbol: position_object}.

        Fetched once per run and passed around so we never call list_positions()
        more than once per session.
        """
        try:
            positions = self.conn.api.list_positions()
            return {p.symbol: p for p in positions}
        except Exception as e:
            logging.error(f"Error fetching held positions: {e}")
            return {}

    def fetch_recent_data(self, symbol, lookback_days=150):
        """Fetch enough history for feature calculation.

        The binary feature builder requires window_size + horizon + 50 bars minimum.
        With window_size=40 and horizon=9, that's 99 trading days (~140 calendar days).
        Default of 150 calendar days provides a comfortable buffer.
        """
        try:
            df = self.conn.get_historical_data(
                symbol,
                days=lookback_days,
                timeframe=TimeFrame.Day
            )
            return df
        except Exception as e:
            logging.error(f"Error fetching recent data for {symbol}: {e}")
            return None

    def check_existing_position(self, symbol, held=None):
        """Check if we already hold a position in this symbol.

        Pass the held dict from get_held_positions() to avoid extra API calls.
        Falls back to a live API call if held is not provided.
        """
        if held is not None:
            return symbol in held
        try:
            positions = self.conn.api.list_positions()
            return any(p.symbol == symbol for p in positions)
        except Exception as e:
            logging.error(f"Error checking positions: {e}")
            return True  # Assume position exists to be safe

    def calculate_position_size(self, symbol, price):
        """Calculate position size based on account and risk management"""
        account = self.conn.get_account()
        buying_power = float(account.buying_power)

        # Risk 2% of account per trade, max $1000
        risk_amount = min(buying_power * 0.02, 1000)
        qty = int(risk_amount / price)

        return max(qty, 1)  # At least 1 share

    def should_buy(self, df):
        """Check if BUY detector triggers for this data."""
        prediction = self.buy_detector.predict(df)
        if prediction and prediction['is_buy']:
            return True, prediction['probability']
        return False, 0.0

    def should_sell(self, df):
        """Check if SELL detector triggers for this data."""
        if self.sell_detector is None:
            return False, 0.0
        prediction = self.sell_detector.predict(df)
        if prediction and prediction['is_sell']:
            return True, prediction['probability']
        return False, 0.0

    def execute_trade(self, symbol, signal, confidence, current_price):
        """Execute trade based on ML signal"""
        if signal == 'BUY':
            qty = self.calculate_position_size(symbol, current_price)

            # Set bracket order prices (2% profit target, 1% stop loss)
            take_profit = round(current_price * 1.02, 2)
            stop_loss = round(current_price * 0.99, 2)

            order = self.conn.place_bracket_order(
                symbol=symbol,
                qty=qty,
                entry_price=current_price,
                take_profit=take_profit,
                stop_loss=stop_loss
            )
            logging.info(f"BUY order placed: {symbol} x{qty} at {current_price}")
            return order

        elif signal == 'SELL':
            try:
                self.conn.api.close_position(symbol)
                logging.info(f"SELL: Closed position in {symbol}")
            except Exception as e:
                logging.error(f"Error closing position for {symbol}: {e}")

        return None

    def run(self, symbols=None, min_confidence=0.6, dry_run=False):
        """Main execution loop.

        Two-pass approach:
          Pass 1 — SELL: iterate over every held position from Alpaca.
                         Ensures we never miss an exit regardless of whether
                         the symbol appears on today's watchlist.
          Pass 2 — BUY:  iterate over the FinViz watchlist, skipping anything
                         already held.
        """
        logging.info("=" * 50)
        logging.info(f"ML Trader started at {datetime.now()}")
        logging.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
        logging.info("=" * 50)

        clock = self.conn.api.get_clock()
        if not clock.is_open:
            logging.warning("Market is closed. Exiting.")
            return []

        # Fetch held positions once — used by both passes
        held = self.get_held_positions()
        logging.info(
            f"Held positions ({len(held)}): "
            f"{', '.join(held.keys()) if held else 'none'}"
        )

        trades_executed = []

        # ------------------------------------------------------------------
        # Pass 1: SELL — check every held position for an exit signal
        # ------------------------------------------------------------------
        logging.info(f"--- SELL pass: {len(held)} held positions ---")

        for symbol, position in held.items():
            try:
                df = self.fetch_recent_data(symbol)
                if df is None or len(df) < 50:
                    logging.warning(f"{symbol}: insufficient data for SELL check")
                    continue

                current_price = float(df['close'].iloc[-1])
                should_sell, confidence = self.should_sell(df)

                logging.info(
                    f"{symbol}: SELL={'YES' if should_sell else 'NO'} "
                    f"(prob={confidence:.3f}) @ ${current_price:.2f}  "
                    f"[entry=${float(position.avg_entry_price):.2f}  "
                    f"P&L={float(position.unrealized_plpc)*100:+.1f}%]"
                )

                if should_sell:
                    if dry_run:
                        logging.info(f"[DRY RUN] Would SELL {symbol}")
                    else:
                        order = self.execute_trade(symbol, 'SELL', confidence, current_price)
                        if order:
                            trades_executed.append({
                                'symbol': symbol,
                                'signal': 'SELL',
                                'confidence': confidence,
                                'price': current_price,
                                'time': datetime.now(),
                            })

            except Exception as e:
                logging.error(f"Error in SELL pass for {symbol}: {e}")
                continue

        # ------------------------------------------------------------------
        # Pass 2: BUY — scan watchlist, skip anything already held
        # ------------------------------------------------------------------
        if symbols is None:
            symbols = self.get_watchlist()

        buy_candidates = [s for s in symbols if s not in held]
        logging.info(f"--- BUY pass: {len(buy_candidates)} candidates "
                     f"({len(symbols) - len(buy_candidates)} skipped — already held) ---")

        for symbol in buy_candidates:
            try:
                df = self.fetch_recent_data(symbol)
                if df is None or len(df) < 50:
                    logging.warning(f"{symbol}: insufficient data for BUY check")
                    continue

                current_price = float(df['close'].iloc[-1])
                should_buy, confidence = self.should_buy(df)

                logging.info(
                    f"{symbol}: BUY={'YES' if should_buy else 'NO'} "
                    f"(prob={confidence:.3f}) @ ${current_price:.2f}"
                )

                if should_buy and confidence >= min_confidence:
                    if dry_run:
                        logging.info(f"[DRY RUN] Would BUY {symbol}")
                    else:
                        order = self.execute_trade(symbol, 'BUY', confidence, current_price)
                        if order:
                            trades_executed.append({
                                'symbol': symbol,
                                'signal': 'BUY',
                                'confidence': confidence,
                                'price': current_price,
                                'time': datetime.now(),
                            })

            except Exception as e:
                logging.error(f"Error in BUY pass for {symbol}: {e}")
                continue

        # Summary
        logging.info("=" * 50)
        logging.info(f"Completed. Trades executed: {len(trades_executed)}")
        for trade in trades_executed:
            logging.info(f"  {trade['signal']} {trade['symbol']} @ ${trade['price']:.2f}")

        return trades_executed


def main():
    parser = argparse.ArgumentParser(description="ML-based stock trader")
    parser.add_argument('--dry-run', action='store_true', help='Preview without trading')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to analyze')
    parser.add_argument('--confidence', type=float, default=0.6, help='Minimum confidence threshold')
    parser.add_argument('--model', default=None, help='Path to BUY champion .pkl (default: latest)')
    parser.add_argument('--sell-model', default=None, help='Path to SELL champion .pkl (default: latest)')
    parser.add_argument('--live', action='store_true', help='Use live trading (default: paper)')

    args = parser.parse_args()

    trader = MLTrader(model_path=args.model, sell_model_path=args.sell_model, paper=not args.live)
    trader.run(
        symbols=args.symbols,
        min_confidence=args.confidence,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
