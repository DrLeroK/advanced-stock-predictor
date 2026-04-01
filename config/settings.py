"""
Configuration settings for the advanced stock predictor
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models' / 'saved'
LOG_DIR = BASE_DIR / 'logs'

# Create directories
for dir_path in [DATA_DIR / 'raw', DATA_DIR / 'processed', DATA_DIR / 'fundamentals',
                 MODEL_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model parameters
class ModelConfig:
    # LSTM parameters
    SEQUENCE_LENGTH = 60  # Days of history to use
    LSTM_UNITS = 128
    DENSE_UNITS = 64
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 0.001
    
    # Classification thresholds
    UP_THRESHOLD = 0.005  # 0.5% for "Up" class
    DOWN_THRESHOLD = -0.005  # -0.5% for "Down" class
    
    # Volatility thresholds
    HIGH_VOL_THRESHOLD = 0.02  # 2% daily return standard deviation
    
    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 32
    PATIENCE = 10
    
    # Ensemble weights
    ENSEMBLE_WEIGHTS = {
        'classifier': 0.4,
        'regressor': 0.3,
        'volatility': 0.3
    }

# Feature columns
FEATURE_COLUMNS = [
    'sma_5', 'sma_10', 'sma_20', 'sma_50',
    'ema_5', 'ema_10', 'ema_20', 'ema_50',
    'return_1d', 'return_5d', 'return_20d',
    'log_return_1d', 'volatility', 'volume_ratio',
    'macd', 'macd_signal', 'macd_diff',
    'rsi', 'bb_position', 'atr_pct',
    'obv', 'mfi'
]

# Fundamentals to fetch
FUNDAMENTAL_COLUMNS = [
    'pe_ratio', 'eps', 'market_cap', 'dividend_yield',
    'revenue_growth', 'earnings_growth', 'profit_margin'
]

# Stock symbols
DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']