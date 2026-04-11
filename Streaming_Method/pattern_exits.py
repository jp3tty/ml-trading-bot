"""
Rule-based exits using TechnicalAnalysis summaries (pattern path).

Use with AlpacaConnection from alpaca_trading (REST only is enough).
"""

import logging

from alpaca_trade_api.rest import TimeFrame

from techAnalysis import TechnicalAnalysis

logger = logging.getLogger(__name__)


class PatternExitManager:
    def __init__(self, conn):
        self.conn = conn
        self.ta = TechnicalAnalysis()

    def should_sell(self, analysis: dict) -> bool:
        engulfing = analysis.get("engulfing_latest")
        bearish_mom = analysis.get("bearish_momentum")
        doji = analysis.get("doji_type")

        if engulfing == "bearish":
            return True
        if bearish_mom:
            return True
        if doji == "gravestone":
            return True

        return False

    def run_pattern_based_exits(self, days: int = 90):
        """Close positions that meet pattern-based SELL rules."""
        logger.info("Running pattern-based exit scan for open positions")

        try:
            positions = self.conn.api.list_positions()
        except Exception as e:
            logger.error("Error fetching positions: %s", e)
            return

        if not positions:
            logger.info("No open positions found")
            return

        for pos in positions:
            symbol = pos.symbol
            logger.info("Analyzing open position: %s", symbol)

            try:
                df = self.conn.get_historical_data(
                    symbol, days=days, timeframe=TimeFrame.Day
                )
                if df.empty or len(df) < 50:
                    logger.warning("Insufficient data for %s, skipping", symbol)
                    continue

                analysis = self.ta.analyze_stock(df, symbol)

                if self.should_sell(analysis):
                    logger.info("SELL signal for %s, closing position", symbol)
                    try:
                        self.conn.api.close_position(symbol)
                        logger.info("Closed position in %s", symbol)
                    except Exception as e:
                        logger.error("Error closing %s: %s", symbol, e)
                else:
                    logger.info("No SELL signal for %s", symbol)

            except Exception as e:
                logger.error("Error analyzing position %s: %s", symbol, e)
