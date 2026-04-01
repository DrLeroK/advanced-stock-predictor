"""
Enhanced technical indicators with more features
"""
import pandas as pd
import numpy as np

def add_advanced_technical_indicators(df):
    """
    Add comprehensive technical indicators
    """
    df = df.copy()
    
    if len(df) < 50:
        return df
    
    # Moving averages with different windows
    for window in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{window}'] = df['close'].rolling(window, min_periods=1).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False, min_periods=1).mean()
    
    # Returns (linear and log)
    for period in [1, 5, 10, 20]:
        df[f'return_{period}d'] = df['close'].pct_change(period).fillna(0)
        df[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period)).fillna(0)
    
    # Volatility measures
    df['volatility_5d'] = df['return_1d'].rolling(5, min_periods=1).std()
    df['volatility_20d'] = df['return_1d'].rolling(20, min_periods=1).std()
    df['volatility_60d'] = df['return_1d'].rolling(60, min_periods=1).std()
    df['volatility_ratio'] = df['volatility_20d'] / df['volatility_60d']
    
    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(20, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ema'] = df['obv'].ewm(span=20, min_periods=1).mean()
    
    # Money Flow Index
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
    money_ratio = positive_flow / negative_flow.replace(0, np.nan)
    df['mfi'] = 100 - (100 / (1 + money_ratio)).fillna(50)
    
    # MACD
    df['ema_12'] = df['close'].ewm(span=12, min_periods=1).mean()
    df['ema_26'] = df['close'].ewm(span=26, min_periods=1).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, min_periods=1).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs)).fillna(50)
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20, min_periods=1).mean()
    bb_std = df['close'].rolling(20, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).fillna(0.5)
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14, min_periods=1).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100
    
    # Price channels
    df['highest_20'] = df['high'].rolling(20, min_periods=1).max()
    df['lowest_20'] = df['low'].rolling(20, min_periods=1).min()
    df['price_position'] = (df['close'] - df['lowest_20']) / (df['highest_20'] - df['lowest_20'])
    
    # Price momentum
    df['momentum'] = df['close'] - df['close'].shift(10)
    df['momentum_pct'] = df['momentum'] / df['close'].shift(10) * 100
    
    # Rate of Change
    for period in [5, 10, 20]:
        df[f'roc_{period}'] = df['close'].pct_change(period) * 100
    
    # Stochastic Oscillator
    low_14 = df['low'].rolling(14, min_periods=1).min()
    high_14 = df['high'].rolling(14, min_periods=1).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14)).fillna(50)
    df['stoch_d'] = df['stoch_k'].rolling(3, min_periods=1).mean()
    
    # Williams %R
    df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14)).fillna(-50)
    
    # Target variables
    df['target_return'] = df['close'].shift(-1) / df['close'] - 1
    df['target_log_return'] = np.log(df['close'].shift(-1) / df['close'])
    df['target_direction'] = (df['target_return'] > 0).astype(int)
    
    # Multi-horizon targets
    for days in [1, 5, 20]:
        df[f'target_return_{days}d'] = df['close'].shift(-days) / df['close'] - 1
        df[f'target_direction_{days}d'] = (df[f'target_return_{days}d'] > 0).astype(int)
    
    return df