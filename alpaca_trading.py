import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from alpaca_trade_api.stream import Stream
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from dotenv import load_dotenv
from techAnalysis import TechnicalAnalysis
from stock_picker.stock_screener import get_tickers

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

class AlpacaConnection:
    def __init__(self, paper=True):
        # Paper trading setup uses different base URL then real trading (this is for paper trading)
        base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"

        self.api = tradeapi.REST(
            key_id=API_KEY,
            secret_key=SECRET_KEY,
            base_url=base_url
        )
 
        # WebSocket stream for real-time data
        stream_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.stream = Stream(
            API_KEY,
            SECRET_KEY,
            base_url=stream_url,
            data_feed="iex"  # Use "sip" for real-time with paid subscription (iex is free)
        )

        # Technical analysis instance (reused across all bars)
        self.ta = TechnicalAnalysis()

    def get_account(self):
        return self.api.get_account()

    def get_historical_data(self, symbol, days=1, timeframe=TimeFrame.Minute):
        """Get historical bars"""
        end = datetime.now()
        start = end - timedelta(days=days)

        bars = self.api.get_bars(
            symbol,
            timeframe,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            feed="iex"  # Use IEX for free accounts (use "sip" for paid)
        ).df

        return bars

    def place_bracket_order(self, symbol, qty, entry_price, take_profit, stop_loss):
        """Bracket order logic"""
        return self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="limit",
            limit_price=entry_price,
            time_in_force="gtc",
            order_class="bracket",
            take_profit={'limit_price': take_profit},
            stop_loss={'limit_price': stop_loss}
        )

    async def on_bar(self,bar):
        """Call when new bar arrives"""
        print(f"New bar: {bar.symbol} Close: {bar.close}")

        # Convert to dict format for techAnalysis
        bar_data = {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        }

        if self.ta.hammer_detect(bar_data):
            print("Hammer Detected, Placing order.")
            order = self.place_bracket_order(bar.symbol, 1, bar.close, bar.close + 0.4, bar.close - 0.4)
            print(f"Order: {order}")

    def start_streaming(self, symbol="SPY"):
        """Start real-time bar streaming - this blocks and runs continuously"""
        self.stream.subscribe_bars(self.on_bar, symbol)
        print(f"Starting stream for {symbol}...")
        self.stream.run()

    def scan_stocks(self, tickers, days=90):
        """
        Scan multiple stocks and return technical analysis results.
        
        Args:
            tickers: List of ticker symbols
            days: Number of days of historical data to analyze
            
        Returns:
            DataFrame with analysis results for each ticker
        """
        results = []
        
        logging.info(f"Scanning {len(tickers)} tickers with {days} days of data")
        
        for i, ticker in enumerate(tickers):
            try:
                logging.info(f"Analyzing {ticker} ({i+1}/{len(tickers)})")
                
                # Fetch historical data
                df = self.get_historical_data(ticker, days=days, timeframe=TimeFrame.Day)
                
                if df.empty:
                    logging.warning(f"No data for {ticker}")
                    continue
                
                # Run analysis
                analysis = self.ta.analyze_stock(df, ticker)
                results.append(analysis)
                
            except Exception as e:
                logging.error(f"Error analyzing {ticker}: {str(e)}")
                results.append({'ticker': ticker, 'error': str(e)})
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        return results_df

def main():
    """Main function to run the stock scanner"""
    logging.info("Starting Stock Scanner")
    
    # Initialize connection
    conn = AlpacaConnection(paper=True)
    
    # Check account
    account = conn.get_account()
    logging.info(f"Account Cash: ${account.cash}")
    
    # Get tickers from FinViz screener
    logging.info("Fetching tickers from FinViz...")
    tickers = get_tickers()
    
    if not tickers:
        logging.error("No tickers found from screener")
        return
    
    logging.info(f"Found {len(tickers)} tickers to analyze")
    
    # Run technical analysis scan (90 days of data)
    results_df = conn.scan_stocks(tickers, days=90)

    if 'error' in results_df.columns:
        results_df = results_df[results_df['error'].isna()]

    # Map to column names
    output_df = pd.DataFrame({
    'Ticker': results_df['ticker'],
    'Latest Price': results_df['price'],
    'Engulfing Signal': results_df['engulfing_latest'].fillna('Neutral'),
    'Momentum Trend': results_df.apply(
        lambda row: 'Bullish' if row['bullish_momentum'] else ('Bearish' if row['bearish_momentum'] else 'Neutral'), 
        axis=1
    ),
    'Hammer Signal': results_df['hammer_recent'].apply(lambda x: f'{int(x)} recent' if x > 0 else ''),
    'Doji Signal': results_df['doji_type'].fillna('Neutral')})

    output_df.to_csv('saved_data/scan_results.csv', index=False)
    logging.info("Results saved to saved_data/scan_results.csv")
    
    logging.info("Stock Scanner completed")


if __name__ == "__main__":
    main()