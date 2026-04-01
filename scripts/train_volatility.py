#!/usr/bin/env python
"""
Train standalone volatility prediction model
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import ModelConfig, FEATURE_COLUMNS
from data.store import AdvancedDataStore
from features.technical import add_advanced_technical_indicators
from features.volatility import add_volatility_features
from features.fundamentals import add_fundamental_features
from features.scalers import RobustTimeSeriesScaler
from features.sequences import create_sequences
from models.volatility_model import VolatilityPredictor

def main():
    parser = argparse.ArgumentParser(description="Train volatility predictor")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lstm-units", type=int, default=64)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🌊 VOLATILITY PREDICTOR TRAINING")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"LSTM Units: {args.lstm_units}")
    print(f"Epochs: {args.epochs}")
    print("=" * 80)
    
    # Load data
    print("\n📂 Loading data...")
    store = AdvancedDataStore()
    df = store.load_raw(args.symbol)
    
    if df is None:
        print(f"❌ No data found for {args.symbol}")
        return
    
    print(f"   ✅ Loaded {len(df)} rows")
    
    # Add technical indicators
    print("\n📊 Adding technical indicators...")
    df = add_advanced_technical_indicators(df)
    print(f"   ✅ Added technical indicators")
    
    # Add volatility features
    print("\n🌊 Adding volatility features...")
    df = add_volatility_features(df)
    print(f"   ✅ Added volatility features")
    
    # Load fundamentals
    print("\n📈 Loading fundamental data...")
    fundamentals = store.load_fundamentals(args.symbol)
    if fundamentals:
        df = add_fundamental_features(df, fundamentals)
        print(f"   ✅ Added fundamentals")
    
    # Prepare features
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    vol_features = ['realized_vol_20d', 'vol_ratio_20_60', 'parkinson_vol_20d']
    available_features.extend([f for f in vol_features if f in df.columns])
    
    fund_features = [col for col in df.columns if col.startswith('fund_')]
    available_features.extend(fund_features)
    
    print(f"   📊 Using {len(available_features)} features")
    
    # Target: next day's realized volatility
    df['target_volatility'] = df['return_1d'].shift(-1).rolling(5, min_periods=1).std() * np.sqrt(252)
    
    # Drop NaN
    df_clean = df[['date'] + available_features + ['target_volatility']].copy()
    df_clean = df_clean.dropna()
    
    print(f"   ✅ Cleaned data: {len(df_clean)} samples")
    print(f"   📊 Target volatility range: {df_clean['target_volatility'].min():.2%} to {df_clean['target_volatility'].max():.2%}")
    
    # Scale features
    print("\n📐 Scaling features...")
    feature_scaler = RobustTimeSeriesScaler()
    X = df_clean[available_features].values.astype(np.float32)
    feature_scaler.fit(X)
    X_scaled = feature_scaler.transform(X)
    
    store.save_scaler(feature_scaler, args.symbol, "volatility_features")
    
    # Create sequences
    print("\n🔗 Creating sequences...")
    X_seq, y_seq = create_sequences(X_scaled, df_clean['target_volatility'].values, args.sequence_length)
    
    print(f"   ✅ Created {len(X_seq)} sequences")
    
    # Train/validation split
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
    
    # Build and train model
    print("\n🎓 Building volatility predictor...")
    volatility_model = VolatilityPredictor(
        sequence_length=args.sequence_length,
        n_features=len(available_features),
        lstm_units=args.lstm_units,
        dropout_rate=0.3,
        learning_rate=0.001
    )
    volatility_model.build()
    volatility_model.model.summary()
    
    print("\n🏋️ Training...")
    history = volatility_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    model_path = f"models/saved/{args.symbol}_volatility.keras"
    volatility_model.model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Evaluate
    print("\n📊 Final Evaluation:")
    y_pred = volatility_model.predict(X_val)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    print(f"   MAE: {mae:.4f} ({mae*100:.2f}%)")
    print(f"   RMSE: {rmse:.4f} ({rmse*100:.2f}%)")
    print(f"   R²: {r2:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
