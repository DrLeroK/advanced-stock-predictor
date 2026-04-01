#!/usr/bin/env python
"""
Train the complete ensemble model - SAVES FEATURE COUNT
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import ModelConfig, FEATURE_COLUMNS
from data.store import AdvancedDataStore
from features.technical import add_advanced_technical_indicators
from features.volatility import add_volatility_features
from features.fundamentals import add_fundamental_features
from features.scalers import RobustTimeSeriesScaler
from features.sequences import create_sequences
from models.classifier import DirectionClassifier
from models.regressor import HuberRegressor
from models.volatility_model import VolatilityPredictor
from models.ensemble import EnsemblePredictor
from models.backtest import AdvancedBacktester

def main():
    parser = argparse.ArgumentParser(description="Train ensemble model")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 ADVANCED ENSEMBLE TRAINING")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Sequence Length: {args.sequence_length}")
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
    print(f"   📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Add technical indicators
    print("\n📊 Adding technical indicators...")
    df = add_advanced_technical_indicators(df)
    print(f"   ✅ Added technical indicators")
    
    # Add volatility features
    print("\n🌊 Adding volatility features...")
    df = add_volatility_features(df)
    print(f"   ✅ Added volatility features")
    
    # Load fundamentals (optional)
    print("\n📈 Loading fundamental data...")
    fundamentals = store.load_fundamentals(args.symbol)
    if fundamentals:
        df = add_fundamental_features(df, fundamentals)
        print(f"   ✅ Added fundamentals")
    else:
        print(f"   ⚠️ No fundamentals found")
    
    # Prepare features
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    
    # Add fundamental features if available
    fund_features = [col for col in df.columns if col.startswith('fund_')]
    available_features.extend(fund_features)
    
    # Add volatility features
    vol_features = ['realized_vol_20d', 'vol_ratio_20_60', 'parkinson_vol_20d']
    available_features.extend([f for f in vol_features if f in df.columns])
    
    print(f"   📊 Using {len(available_features)} features")
    
    # SAVE FEATURE LIST FOR LATER
    feature_file = f"models/saved/{args.symbol}_features.txt"
    with open(feature_file, 'w') as f:
        f.write('\n'.join(available_features))
    print(f"   💾 Saved feature list to {feature_file}")
    
    # Create target variables
    df['target_direction'] = (df['target_return'] > ModelConfig.UP_THRESHOLD).astype(int)
    df['target_return_log'] = np.log1p(df['target_return'])
    
    # Drop NaN
    df_clean = df[['date'] + available_features + 
                  ['target_direction', 'target_return', 'target_return_log', 
                   'target_volatility', 'high_volatility']].copy()
    df_clean = df_clean.dropna()
    
    print(f"   ✅ Cleaned data: {len(df_clean)} samples")
    
    # Scale features
    print("\n📐 Scaling features...")
    feature_scaler = RobustTimeSeriesScaler()
    X = df_clean[available_features].values
    feature_scaler.fit(X)
    X_scaled = feature_scaler.transform(X)
    
    store.save_scaler(feature_scaler, args.symbol, "features")
    
    # Create sequences
    print("\n🔗 Creating sequences...")
    X_seq, y_dir = create_sequences(X_scaled, df_clean['target_direction'].values, args.sequence_length)
    _, y_ret = create_sequences(X_scaled, df_clean['target_return'].values, args.sequence_length)
    _, y_vol = create_sequences(X_scaled, df_clean['target_volatility'].values, args.sequence_length)
    
    print(f"   ✅ Created {len(X_seq)} sequences")
    
    # Train/validation split
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_dir_train, y_dir_val = y_dir[:split_idx], y_dir[split_idx:]
    y_ret_train, y_ret_val = y_ret[:split_idx], y_ret[split_idx:]
    y_vol_train, y_vol_val = y_vol[:split_idx], y_vol[split_idx:]
    
    # Train models
    print("\n🎓 Training Direction Classifier...")
    classifier = DirectionClassifier(
        sequence_length=args.sequence_length,
        n_features=len(available_features),
        lstm_units=128,
        dropout_rate=0.3
    )
    classifier.build()
    classifier.train(X_train, y_dir_train, X_val, y_dir_val, 
                     epochs=args.epochs, batch_size=args.batch_size)
    classifier.model.save(f"models/saved/{args.symbol}_classifier.keras")
    
    print("\n🎓 Training Return Regressor...")
    regressor = HuberRegressor(
        sequence_length=args.sequence_length,
        n_features=len(available_features),
        lstm_units=128,
        dropout_rate=0.3,
        huber_delta=0.5
    )
    regressor.build()
    regressor.train(X_train, y_ret_train, X_val, y_ret_val,
                    epochs=args.epochs, batch_size=args.batch_size)
    regressor.model.save(f"models/saved/{args.symbol}_regressor.keras")
    
    print("\n🎓 Training Volatility Predictor...")
    volatility_model = VolatilityPredictor(
        sequence_length=args.sequence_length,
        n_features=len(available_features),
        lstm_units=64,
        dropout_rate=0.3
    )
    volatility_model.build()
    volatility_model.train(X_train, y_vol_train, X_val, y_vol_val,
                           epochs=args.epochs, batch_size=args.batch_size)
    volatility_model.model.save(f"models/saved/{args.symbol}_volatility.keras")
    
    # Create ensemble
    print("\n🔗 Creating Ensemble...")
    ensemble = EnsemblePredictor(weights={'classifier': 0.4, 'regressor': 0.3, 'volatility': 0.3})
    ensemble.fit(classifier, regressor, volatility_model)
    
    # Run backtest
    print("\n🔬 Running Walk-Forward Backtest...")
    backtester = AdvancedBacktester()
    results, metrics = backtester.backtest(
        df_clean, available_features, args.sequence_length,
        args.symbol, ensemble, feature_scaler
    )
    
    print("\n" + "=" * 80)
    print("📊 BACKTEST RESULTS")
    print("=" * 80)
    print(f"Total Predictions: {metrics['total_predictions']}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print("=" * 80)
    
    # Save results
    results.to_csv(f"backtest_results_{args.symbol}_ensemble.csv", index=False)
    with open(f"backtest_metrics_{args.symbol}_ensemble.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n💾 Results saved to:")
    print(f"   backtest_results_{args.symbol}_ensemble.csv")
    print(f"   backtest_metrics_{args.symbol}_ensemble.json")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print("\n💾 Models saved to: models/saved/")
    print(f"\n🔮 To make predictions:")
    print(f"   python scripts/predict.py --symbol {args.symbol}")

if __name__ == "__main__":
    main()
