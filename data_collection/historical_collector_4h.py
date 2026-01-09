import pandas as pd
from datetime import datetime, timedelta
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
import time
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class HistoricalDataCollector4H:
    """Collector for 4-hour candlestick data"""


    def __init__(self, alpaca_conn, data_dir="saved_data/historical_4h"):
        self.api = alpaca_conn.api
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.WINDOWS_RESERVED = {'CON', 'PRN', 'AUX', 'NUL',
                            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
                            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}

    def get_all_tradeable_symbols(self, min_price=5, max_price=500):
        """Get all active, tradeable US equities from Alpaca"""
        assets = self.api.list_assets(status='active', asset_class='us_equity')
        symbols = [
            a.symbol for a in assets
            if a.tradable and a.shortable  # liquid stocks
            and not a.symbol.isdigit()     # filter out weird tickers
            and '.' not in a.symbol        # no preferred shares
        ]
        return symbols

    def fetch_and_save_historical(self, symbols, start_date, end_date, batch_size=200):
        """
        Fetch 4-hour historical data for many symbols in batches.

        note: Alpaca intraday data typically goes back ~5-7 years for paid plans,
        less for free IEX feed. Adjust start_date accordingly.
        """
        all_data = {}
        
        # 4-hour timeframe
        timeframe = TimeFrame(4, TimeFrameUnit.Hour)

        total_batches = (len(symbols) - 1) // batch_size + 1

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            batch_num = i // batch_size + 1
            logging.info(f"Fetching batch {batch_num}/{total_batches} ({len(batch)} symbols)")

            try:
                # Multi-symbol request (faster than one-by-one)
                bars = self.api.get_bars(
                    batch,
                    timeframe,
                    start=start_date,
                    end=end_date,
                    feed="iex",
                    adjustment='all'
                ).df

                if not bars.empty:
                    if 'symbol' in bars.index.names:
                        for symbol in bars.index.get_level_values('symbol').unique():
                            symbol_data = bars.xs(symbol, level='symbol')
                            all_data[symbol] = symbol_data
                    elif 'symbol' in bars.columns:
                        for symbol in bars['symbol'].unique():
                            symbol_data = bars[bars['symbol'] == symbol].drop(columns=['symbol'])
                            all_data[symbol] = symbol_data
            
            except Exception as e:
                logging.error(f"Error fetching batch {batch_num}: {e}")

            time.sleep(0.5)  # Rate limiting

        # Save as parquet
        logging.info(f"Saving {len(all_data)} symbols to {self.data_dir}...")
        skipped = []
        for symbol, df in all_data.items():
            # Skipping Windows reserved filenames
            if symbol.upper() in self.WINDOWS_RESERVED:
                skipped.append(symbol)
                continue
            df.to_parquet(f"{self.data_dir}/{symbol}.parquet")

        if skipped:
            logging.warning(f"Skipped {len(skipped)} symbols due to Windows reserved names: {skipped}")
            
        logging.info(f"Successfully saved {len(all_data) - len(skipped)} symbols")
        return all_data

def main():
    """Main function to collect historical data"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from alpaca_trading import AlpacaConnection

    logging.info("Starting 4-hourHistorical Data Collection")
    
    conn = AlpacaConnection(paper=True)
    collector = HistoricalDataCollector4H(conn)

    # Get tradeable symbols
    symbols = collector.get_all_tradeable_symbols()
    logging.info(f"Found {len(symbols)} tradeable symbols")

    # Fetch 2 years of 4-hour data
    collector.fetch_and_save_historical(
        symbols[:500],  # Start with top 500
        start_date="2022-01-01",
        end_date="2024-12-01",
        batch_size=200
    )
    
    logging.info("4-hour Historical Data Collection completed")

if __name__ == "__main__":
    main()