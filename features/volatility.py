"""
Volatility prediction features
"""
import pandas as pd
import numpy as np

def add_volatility_features(df):
    """
    Add features for volatility prediction
    """
    df = df.copy()
    
    # Realized volatility
    df['realized_vol_5d'] = df['return_1d'].rolling(5, min_periods=1).std() * np.sqrt(252)
    df['realized_vol_20d'] = df['return_1d'].rolling(20, min_periods=1).std() * np.sqrt(252)
    df['realized_vol_60d'] = df['return_1d'].rolling(60, min_periods=1).std() * np.sqrt(252)
    
    # Volatility ratio
    df['vol_ratio_20_60'] = df['realized_vol_20d'] / df['realized_vol_60d']
    
    # Volatility regime
    df['high_volatility'] = (df['realized_vol_20d'] > 0.25).astype(int)  # 25% annualized = high vol
    
    # Parkinson volatility (high-low range)
    df['parkinson_vol'] = np.sqrt(1 / (4 * np.log(2)) * 
                                   (np.log(df['high'] / df['low']) ** 2))
    df['parkinson_vol_20d'] = df['parkinson_vol'].rolling(20, min_periods=1).mean()
    
    # Garman-Klass volatility
    df['gk_vol'] = np.sqrt(0.5 * (np.log(df['high'] / df['low']) ** 2) - 
                           (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2))
    df['gk_vol_20d'] = df['gk_vol'].rolling(20, min_periods=1).mean()
    
    # Target: tomorrow's realized volatility
    df['target_volatility'] = df['return_1d'].shift(-1).rolling(5, min_periods=1).std() * np.sqrt(252)
    df['target_high_vol'] = (df['target_volatility'] > 0.25).astype(int)
    
    return df