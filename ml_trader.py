import argparse
import csv
import logging
import os
import sys
from datetime import datetime

import pandas as pd
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit

from alpaca_trading import AlpacaConnection
from ml.binary_predictor import BinaryBuyPredictor
from ml.binary_sell_predictor import BinarySellPredictor
from techAnalysis import TechnicalAnalysis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MAX_POSITIONS = 20

ORDER_LOG    = "paper_trade_log/orders.csv"
ORDER_FIELDS = [
    'timestamp', 'symbol', 'side', 'qty', 'entry_price',
    'take_profit', 'stop_loss', 'order_id', 'confidence', 'rsi', 'momentum',
]

_ta = TechnicalAnalysis()

STOP_LOSS_PCT   = 0.020   # -2.0%: wider than training labels to survive intraday noise on 4h bars
USE_TAKE_PROFIT = True    # +2.5%: maintains 1.25× R/R ratio with wider stop
TAKE_PROFIT_PCT = 0.025
MIN_SELL_PNL_PCT = 0.005  # +0.5%: ML SELL won't fire below this gain (filters noise)


def _get_indicators(df):
    try:
        enhanced = _ta.momentum_trend(df.copy())
        last = enhanced.iloc[-1]
        return {
            'rsi':      round(float(last['rsi']), 1),
            'momentum': round(float(last['momentum_strength']), 4),
        }
    except Exception:
        return {'rsi': None, 'momentum': None}


def _log_order(symbol, side, qty, entry_price, take_profit, stop_loss,
               order_id, confidence, rsi, momentum):
    row = {
        'timestamp':   datetime.now().isoformat(),
        'symbol':      symbol,
        'side':        side,
        'qty':         qty,
        'entry_price': entry_price,
        'take_profit': take_profit,
        'stop_loss':   stop_loss,
        'order_id':    order_id,
        'confidence':  round(confidence, 4),
        'rsi':         rsi,
        'momentum':    momentum,
    }
    file_exists = os.path.isfile(ORDER_LOG) and os.path.getsize(ORDER_LOG) > 0
    with open(ORDER_LOG, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ORDER_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

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

    def fetch_recent_data(self, symbol, lookback_days=90):
        """Fetch 4-hour bars to match the timeframe the models were trained on.

        Training used saved_data/historical_4h (TimeFrame 4h). Live inference
        must use the same timeframe or the features are meaningless.
        90 calendar days → ~120 four-hour bars, above the 92-bar minimum
        (window_size=30 + horizon=12 + 50 buffer).
        """
        try:
            df = self.conn.get_historical_data(
                symbol,
                days=lookback_days,
                timeframe=TimeFrame(4, TimeFrameUnit.Hour),
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

    def execute_trade(self, symbol, signal, confidence, current_price,
                      stop_loss=None, take_profit=None):
        """Execute trade based on ML signal"""
        if signal == 'BUY':
            qty         = self.calculate_position_size(symbol, current_price)
            entry_price = round(current_price, 2)

            order = self.conn.place_bracket_order(
                symbol=symbol,
                qty=qty,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            tp_str = f"  tp=${take_profit:.2f}" if take_profit else ""
            logging.info(
                f"BUY order placed: {symbol} x{qty} @ ${current_price:.2f} | "
                f"stop=${stop_loss:.2f}{tp_str}"
            )
            return order

        elif signal == 'SELL':
            try:
                order = self.conn.api.close_position(symbol)
                logging.info(f"SELL: Closed position in {symbol}")
                return order
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
        mode = 'DRY RUN' if dry_run else ('PAPER' if self.conn.paper else 'LIVE')
        logging.info(f"Mode: {mode}")
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
                if df is None or len(df) < 100:
                    logging.warning(f"{symbol}: insufficient data for SELL check")
                    continue

                current_price = float(df['close'].iloc[-1])
                should_sell, confidence = self.should_sell(df)

                avg_entry = float(position.avg_entry_price)
                pnl_pct = (current_price - avg_entry) / avg_entry
                logging.info(
                    f"{symbol}: SELL={'YES' if should_sell else 'NO'} "
                    f"(prob={confidence:.3f}) @ ${current_price:.2f}  "
                    f"[entry=${avg_entry:.2f}  "
                    f"P&L={pnl_pct*100:+.1f}%]"
                )

                if should_sell and pnl_pct < MIN_SELL_PNL_PCT:
                    logging.info(
                        f"{symbol}: SELL gated — P&L {pnl_pct*100:+.2f}% "
                        f"below minimum +{MIN_SELL_PNL_PCT*100:.1f}%"
                    )
                    should_sell = False

                if should_sell:
                    if dry_run:
                        logging.info(f"[DRY RUN] Would SELL {symbol}")
                    else:
                        order = self.execute_trade(symbol, 'SELL', confidence, current_price)
                        if order:
                            ind = _get_indicators(df)
                            _log_order(
                                symbol=symbol, side='SELL',
                                qty=float(getattr(position, 'qty', 0)),
                                entry_price=current_price,
                                take_profit=0, stop_loss=0,
                                order_id=getattr(order, 'id', 'N/A'),
                                confidence=confidence,
                                rsi=ind['rsi'], momentum=ind['momentum'],
                            )
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
        open_slots     = max(0, MAX_POSITIONS - len(held))
        logging.info(
            f"--- BUY pass: {len(buy_candidates)} candidates "
            f"({len(symbols) - len(buy_candidates)} skipped — already held) | "
            f"position slots available: {open_slots}/{MAX_POSITIONS} ---"
        )

        if open_slots == 0:
            logging.info("  At position limit — BUY pass skipped.")
            buy_candidates = []

        # Step 1: Score all candidates
        scored_candidates = []
        for symbol in buy_candidates:
            try:
                df = self.fetch_recent_data(symbol)
                if df is None or len(df) < 100:
                    logging.warning(f"{symbol}: insufficient data for BUY check")
                    continue

                current_price = float(df['close'].iloc[-1])
                prediction    = self.buy_detector.predict(df)
                if prediction is None:
                    continue

                scored_candidates.append((symbol, prediction, current_price, df))
                logging.info(
                    f"{symbol}: scored  BUY={'YES' if prediction['is_buy'] else 'NO'} "
                    f"(prob={prediction['probability']:.3f}) @ ${current_price:.2f}"
                )

            except Exception as e:
                logging.error(f"Error in BUY pass for {symbol}: {e}")
                continue

        # Step 2: Rank by probability, keep top N (capped by open slots and 5-per-run limit)
        scored_candidates.sort(key=lambda x: x[1]['probability'], reverse=True)
        max_buys = min(5, open_slots)
        top5     = scored_candidates[:max_buys]
        logging.info(
            f"  Top {max_buys} of {len(scored_candidates)} scored: "
            f"{[s for s, _, _, _ in top5]}"
        )

        # Step 3: Act on top 5 only
        for symbol, prediction, current_price, df in top5:
            confidence = prediction['probability']
            should_buy = prediction['is_buy']

            logging.info(
                f"{symbol}: BUY={'YES' if should_buy else 'NO'} "
                f"(prob={confidence:.3f}) @ ${current_price:.2f} [top 5]"
            )

            if should_buy and confidence >= min_confidence:
                # Fetch live ask price so the limit order and ATR stop are anchored
                # to the current market, not yesterday's close.
                live_price = self.conn.get_live_price(symbol)
                if live_price is None:
                    logging.warning(
                        f"{symbol}: live quote unavailable — using prior close ${current_price:.2f}"
                    )
                    live_price = current_price
                else:
                    logging.info(
                        f"{symbol}: live ask=${live_price:.2f}  prior close=${current_price:.2f}"
                    )

                stop_loss   = round(live_price * (1 - STOP_LOSS_PCT), 2)
                take_profit = round(live_price * (1 + TAKE_PROFIT_PCT), 2) if USE_TAKE_PROFIT else None
                logging.info(
                    f"{symbol}: stop=${stop_loss:.2f} (-{STOP_LOSS_PCT*100:.1f}%)  "
                    f"tp=${take_profit:.2f} (+{TAKE_PROFIT_PCT*100:.1f}%)"
                )

                if dry_run:
                    tp_str = f"  tp=${take_profit:.2f}" if take_profit else ""
                    logging.info(f"[DRY RUN] Would BUY {symbol}  stop=${stop_loss:.2f}{tp_str}")
                else:
                    order = self.execute_trade(
                        symbol, 'BUY', confidence, live_price,
                        stop_loss=stop_loss, take_profit=take_profit,
                    )
                    if order:
                        ind = _get_indicators(df)
                        _log_order(
                            symbol=symbol, side='BUY',
                            qty=int(getattr(order, 'qty', 0)),
                            entry_price=live_price,
                            take_profit=take_profit or 0,
                            stop_loss=stop_loss,
                            order_id=getattr(order, 'id', 'N/A'),
                            confidence=confidence,
                            rsi=ind['rsi'], momentum=ind['momentum'],
                        )
                        trades_executed.append({
                            'symbol': symbol,
                            'signal': 'BUY',
                            'confidence': confidence,
                            'price': live_price,
                            'time': datetime.now(),
                        })

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
    parser.add_argument('--confidence', type=float, default=0.45, help='Minimum confidence threshold')
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
