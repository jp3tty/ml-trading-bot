import pandas as pd
import numpy as np

class TechnicalAnalysis:
    """
    Technical analysis class for candlestick patterns and momentum indicators.
    Standard keys (lowercase): open, high, low, close,, volume, date, ticker
    """

    # === SINGLE BAR PATTERN DETECTION ===
    def hammerDetect(self, bar):
        """Detect hammer/inverted hammer patterns on a single bar (dict)"""
        body_size = abs(bar['open'] - bar['close'])

        if bar['open'] > bar['close']:      # Red Candle
            lower_wick = abs(bar['close'] - bar['low'])
            upper_wick = abs(bar['high'] - bar['open'])
        else:                               # Green Candle
            lower_wick = abs(bar['open'] - bar['low'])
            upper_wick = abs(bar['high'] - bar['close'])

        is_hammer = lower_wick >= body_size and upper_wick < body_size * 0.25
        is_inverted = upper_wick >= body_size and lower_wick < body_size * 0.25
            
        return is_hammer or is_inverted

    def doji_detect(self, bar, body_threshold=0.001):
        """
        Detect doji patterns on a single bar (dict).
        
        Doji = open and close are nearly equal (small body relative to range).
        
        Returns: dict with pattern type or None if no doji detected
            - 'standard': Small body, balanced wicks
            - 'long_legged': Small body, long upper and lower wicks
            - 'dragonfly': Small body at top, long lower wick (bullish reversal)
            - 'gravestone': Small body at bottom, long upper wick (bearish reversal)
        """
        body_size = abs(bar['open'] - bar['close'])
        candle_range = bar['high'] - bar['low']
        
        # Avoid division by zero on flat candles
        if candle_range == 0:
            return None
        
        body_ratio = body_size / candle_range
        
        # Doji requires very small body relative to total range
        if body_ratio > body_threshold * 10:  # Body must be < 1% of range by default
            return None
        
        # Calculate wick positions
        body_top = max(bar['open'], bar['close'])
        body_bottom = min(bar['open'], bar['close'])
        upper_wick = bar['high'] - body_top
        lower_wick = body_bottom - bar['low']
        
        upper_ratio = upper_wick / candle_range
        lower_ratio = lower_wick / candle_range
        
        # Classify doji type
        if lower_ratio > 0.6 and upper_ratio < 0.1:
            return {'type': 'dragonfly', 'signal': 'bullish'}
        elif upper_ratio > 0.6 and lower_ratio < 0.1:
            return {'type': 'gravestone', 'signal': 'bearish'}
        elif upper_ratio > 0.3 and lower_ratio > 0.3:
            return {'type': 'long_legged', 'signal': 'neutral'}
        else:
            return {'type': 'standard', 'signal': 'neutral'}

    
    # === DATAFRAME PATTERN DETECTION ===
    def engulfing_signal(self, df, body_diff_min=0.003):
        """
        Detect engulfing patterns across a DataFrame.
        Returns: Series with 0=neutral, 1=bearish engulfing, 2=bullish engulfing
        """
        df = self._normalize_df(df)

        body_curr = abs(df['open'] - df['close'])
        body_prev = body_curr.shift(1)

        prev_bullish = df['open'].shift(1) < df['close'].shift(1)
        prev_bearish = df['open'].shift(1) > df['close'].shift(1)
        curr_bullish = df['open'] > df['close']
        curr_bearish = df['open'] < df['close']

        bearish_engulf = (
            (body_curr > body_diff_min) & (body_prev > body_diff_min) & 
            prev_bullish & curr_bearish &
            (df['open'] >= df['close'].shift(1)) &
            (df['close'] <= df['open'].shift(1))
        )

        bullish_engulf = (
            (body_curr > body_diff_min) & (body_prev > body_diff_min) & 
            prev_bearish & curr_bullish &
            (df['open'] <= df['close'].shift(1)) &
            (df['close'] >= df['open'].shift(1))
        )

        signal = pd.Series(0, index=df.index)
        signal[bearish_engulf] = 1
        signal[bullish_engulf] = 2
        return signal
        
    
    # === MOMENTUM INDICATORS ===
    def rsi(self, series, period=14):
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def momentum(self, series, period=10):
        """Calculate price momentum"""
        return series.diff(period)

    def sma(self, series, period):
        """Simple moving average"""
        return series.rolling(window=period).mean()

    def momentum_trend(self, df):
        """
        Add momentum indicators and trend signals to DataFrame.
        Returns: DataFrame with added columns.
        """
        df = self._normalize_df(df.copy())

        df['rsi'] = self.rsi(df['close'])
        df['momentum'] = self.momentum(df['close'])
        df['sma_20'] = self.sma(df['close'], 20)
        df['sma_50'] = self.sma(df['close'], 50)
        df['momentum_strength'] = df['momentum'] / df['close'] * 100 

        df['bullish_momentum'] = (
            (df['rsi'] > 50) & 
            (df['close'] > df['sma_20']) &
            (df['sma_20'] > df['sma_50']) &
            (df['momentum'] > 0)
        )

        df['bearish_momentum'] = (
            (df['rsi'] < 50) & 
            (df['close'] < df['sma_20']) &
            (df['sma_20'] < df['sma_50']) &
            (df['momentum'] < 0)
        )

        return df


    # === DATAFRAME HAMMER/DOJI DETECTION ===
    def hammer_signal(self, df):
        """
        Detect hammer/inverted hammer patterns across DataFrame.
        Returns: Series with True where hammer detected
        """
        df = self._normalize_df(df.copy())
        
        body_size = abs(df['open'] - df['close'])
        
        # Calculate wicks based on candle color
        is_red = df['open'] > df['close']
        
        lower_wick = np.where(
            is_red,
            abs(df['close'] - df['low']),
            abs(df['open'] - df['low'])
        )
        upper_wick = np.where(
            is_red,
            abs(df['high'] - df['open']),
            abs(df['high'] - df['close'])
        )
        
        is_hammer = (lower_wick >= body_size) & (upper_wick < body_size * 0.25)
        is_inverted = (upper_wick >= body_size) & (lower_wick < body_size * 0.25)
        
        return pd.Series(is_hammer | is_inverted, index=df.index)

    def doji_signal(self, df, body_threshold=0.01):
        """
        Detect doji patterns across DataFrame.
        Returns: Series with doji type string or None
        """
        df = self._normalize_df(df.copy())
        
        body_size = abs(df['open'] - df['close'])
        candle_range = df['high'] - df['low']
        
        # Avoid division by zero
        candle_range = candle_range.replace(0, np.nan)
        body_ratio = body_size / candle_range
        
        # Calculate wick positions
        body_top = df[['open', 'close']].max(axis=1)
        body_bottom = df[['open', 'close']].min(axis=1)
        upper_wick = df['high'] - body_top
        lower_wick = body_bottom - df['low']
        
        upper_ratio = upper_wick / candle_range
        lower_ratio = lower_wick / candle_range
        
        # Doji requires small body
        is_doji = body_ratio < body_threshold
        
        result = pd.Series(None, index=df.index, dtype=object)
        result[is_doji & (lower_ratio > 0.6) & (upper_ratio < 0.1)] = 'dragonfly'
        result[is_doji & (upper_ratio > 0.6) & (lower_ratio < 0.1)] = 'gravestone'
        result[is_doji & (upper_ratio > 0.3) & (lower_ratio > 0.3)] = 'long_legged'
        result[is_doji & result.isna()] = 'standard'
        
        return result

    def analyze_stock(self, df, ticker):
        """
        Complete analysis of a stock DataFrame.
        Returns dict with latest indicator values for results table.
        """
        df = self._normalize_df(df.copy())
        
        if len(df) < 50:
            return {
                'ticker': ticker,
                'error': 'Insufficient data'
            }
        
        # Add momentum indicators
        df = self.momentum_trend(df)
        
        # Add pattern signals
        df['hammer'] = self.hammer_signal(df)
        df['doji'] = self.doji_signal(df)
        df['engulfing'] = self.engulfing_signal(df)
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Count recent patterns (last 5 days)
        recent = df.tail(5)
        
        return {
            'ticker': ticker,
            'price': round(latest['close'], 2),
            'rsi': round(latest['rsi'], 2) if pd.notna(latest['rsi']) else None,
            'momentum': round(latest['momentum'], 4) if pd.notna(latest['momentum']) else None,
            'momentum_pct': round(latest['momentum_strength'], 2) if pd.notna(latest['momentum_strength']) else None,
            'sma_20': round(latest['sma_20'], 2) if pd.notna(latest['sma_20']) else None,
            'sma_50': round(latest['sma_50'], 2) if pd.notna(latest['sma_50']) else None,
            'bullish_momentum': bool(latest['bullish_momentum']),
            'bearish_momentum': bool(latest['bearish_momentum']),
            'hammer_recent': int(recent['hammer'].sum()),
            'doji_recent': recent['doji'].notna().sum(),
            'doji_type': latest['doji'] if pd.notna(latest['doji']) else None,
            'engulfing_latest': 'bullish' if latest['engulfing'] == 2 else ('bearish' if latest['engulfing'] == 1 else None),
            'engulfing_recent': int((recent['engulfing'] > 0).sum())
        }

    # === HELPER METHODS ===
    def _normalize_df(self, df):
        """Normalize column names to lowercase"""
        df.columns = df.columns.str.lower()
        return df