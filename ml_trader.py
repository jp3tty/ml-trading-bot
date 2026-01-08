import argparse
import logging
import os
import sys
from datetime import datetime

from numpy import take_along_axis
import pandas as pd
from alpaca_trade_api.rest import TimeFrame

import pandas as pd
from alpaca_trading import AlpacaConnection
from ml.feature_builder import FeatureBuilder
from ml.predictor import TradingPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MLTrader:
    def __init__(self, model_path='models/rocket_trading_model.pkl', paper=True):
        self.conn = AlpacaConnection(paper=paper)
        self.feature_builder = FeatureBuilder(window_size=20, horizon=5)
        self.predictor = TradingPredictor(model_path, self.feature_builder)

    def get_watchlist(self):
        """Symbols to analyze - currently just all stocks from original FinViz screener"""
        
        from stock_picker.stock_screener import get_tickers
        return get_tickers()

    def fetch_recent_data(self, symbol, lookback_days=30):
        """Fetch enough history for feature calculation"""
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

    def check_existing_position(self, symbol):
        """Check if we already have a position in this symbol"""
        try:
            positions = self.conn.api.list_positions()
            return any(p.symbol == symbol for p in positions)
        except Exception as e:
            logging.error(f"Error checking positions: {e}")
            return True   # Assume position exists to be safe

    def calculate_position_size(self, symbol, price):
        """Calculate position size based on account and risk management"""
        account = self.conn.get_account()
        buying_power = float(account.buying_power)

        # Risk 2% of account per trade, max $1000
        risk_amount = min(buying_power * 0.02, 1000)
        qty = int(risk_amount / price)
    
        return max(qty, 1)   # At least 1 share

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
            # Close existing positiion if we have one
            if self.check_exisiting_position(symbol):
                try:
                    self.conn.api.close_position(symbol)
                    logging.info(f"SELL: Closed position in {symbol}")
                except Exception as e:
                    logging.error(f"Error closing position for {symbol}: {e}")

        return None

    def run(self, symbols=None, min_confidence=0.6, dry_run=False):
        """Main execution loop"""
        logging.info("=" * 50)
        logging.info(f"ML Trader started at {datetime.now()}")
        logging.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
        logging.info(f"=" * 50)

        # Check market hours
        clock = self.conn.api.get_clock()
        if not clock.is_open:
            logging.warning("Market is closed. Exiting.")
            return []

        # Get symbols to analyze
        if symbols is None:
            symbols = self.get_watchlist()

        logging.info(f"Analyzing {len(symbols)} symbols...")

        trades_executed = []

        for symbol in symbols:
            try:
                # skip if we already have a position
                if self.check_existing_position(symbol):
                    logging.info(f"Skipping {symbol} - already in position")
                    continue

                # fetch data
                df = self.fetch_recent_data(symbol)
                if df is None or len(df) < 50:
                    logging.warning(f"Insufficient data for {symbol}")
                    continue

                # get prediction
                prediction = self.predictor.predict(df)
                if prediction is None:
                    continue

                signal = predition['signal']
                confidence = prediction['confidence']
                current_price = df['close'].iloc[-1]

                logging.info(
                    f"{symbol}: {signal} (confidence: {confidence: .2f}) @ ${current_price:.2f}"
                )

                # execute if confidence threshold is met
                if signal in ['BUY', 'SELL'] and confidence >= min_confidence:
                    if dry_run:
                        logging.info(f"[DRY RUN] Would execute {signal} on {symbol}")
                    else:
                        order = self.execute_trade(symbol, signal, confidence, current_price)
                        if order:
                            trades_executed.append({
                                'symbol': symbol,
                                'signal': signal,
                                'confidence': confidence,
                                'price': current_price,
                                'time': datetime.now()
                            })

            except Exception as e:
                logging.error(f"Error processing {symbol}: {e}")
                continue

        # summary
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
    parser.add_argument('--model', default='models/rocket_trading_model.pkl', help='Model path')
    parser.add_argument('--live', action='store_true', help='Use live trading(default: paper)')

    args = parser.parse_args()

    trader = MLTrader(model_path=args.model, paper=not args.live)
    trader.run(
        symbols=args.symbols,
        min_confidence=args.confidence,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()