#!/usr/bin/env python
"""
Improved training script with feature selection and hyperparameter tuning
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
from features.feature_selector import FeatureSelector
from features.scalers import RobustTimeSeriesScaler
from features.sequences import create_sequences
from models.classifier import DirectionClassifier
from models.regressor import HuberRegressor
from models.volatility_model import VolatilityPredictor
from models.ensemble import EnsemblePredictor
from models.backtest import AdvancedBacktester

def main():
    parser = argparse.ArgumentParser(description="Train improved ensemble model")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lstm-units", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--use-feature-selection", action="store_true", 
                       help="Use feature selection to reduce overfitting")
    parser.add_argument("--n-features", type=int, default=20,
                       help="Number of features to keep if using selection")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 IMPROVED ENSEMBLE TRAINING")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Epochs: {args.epochs}")
    print(f"LSTM Units: {args.lstm_units}")
    print(f"Dropout: {args.dropout}")
    print(f"Feature Selection: {'ON' if args.use_feature_selection else 'OFF'}")
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
    df = add_volatility_features(df)
    
    # Load fundamentals (optional)
    print("\n📈 Loading fundamental data...")
    fundamentals = store.load_fundamentals(args.symbol)
    if fundamentals:
        df = add_fundamental_features(df, fundamentals)
        print(f"   ✅ Added fundamentals")
    else:
        print(f"   ⚠️ No fundamentals found")
    
    # Prepare initial features
    all_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    fund_features = [col for col in df.columns if col.startswith('fund_')]
    vol_features = ['realized_vol_20d', 'vol_ratio_20_60', 'parkinson_vol_20d']
    
    available_features = all_features + fund_features + [f for f in vol_features if f in df.columns]
    
    print(f"   📊 Initial features: {len(available_features)}")
    
    # Create target
    df['target_direction'] = (df['target_return'] > ModelConfig.UP_THRESHOLD).astype(int)
    
    # Clean data
    df_clean = df[['date'] + available_features + 
                  ['target_direction', 'target_return', 'target_volatility']].copy()
    df_clean = df_clean.dropna()
    
    print(f"   ✅ Cleaned data: {len(df_clean)} samples")
    
    # Feature selection
    if args.use_feature_selection and len(available_features) > args.n_features:
        print(f"\n🔍 Performing feature selection...")
        
        X_temp = df_clean[available_features].values
        y_temp = df_clean['target_direction'].values
        
        # Create DataFrame for feature names
        X_df = pd.DataFrame(X_temp, columns=available_features)
        
        # Select top features
        selected_features, selector = FeatureSelector.select_top_features(
            X_df, y_temp, k=args.n_features
        )
        
        available_features = selected_features
        print(f"   ✅ Reduced to {len(available_features)} features")
        
        # Save selected features
        with open(f"models/saved/{args.symbol}_selected_features.txt", 'w') as f:
            f.write('\n'.join(selected_features))
    else:
        # Save full feature list
        with open(f"models/saved/{args.symbol}_features.txt", 'w') as f:
            f.write('\n'.join(available_features))
    
    print(f"   📊 Final features: {len(available_features)}")
    
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
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate
    )
    classifier.build()
    classifier.train(X_train, y_dir_train, X_val, y_dir_val, 
                     epochs=args.epochs, batch_size=args.batch_size)
    classifier.model.save(f"models/saved/{args.symbol}_classifier.keras")
    
    print("\n🎓 Training Return Regressor...")
    regressor = HuberRegressor(
        sequence_length=args.sequence_length,
        n_features=len(available_features),
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate,
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
        lstm_units=args.lstm_units // 2,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate
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
    print("📊 IMPROVED BACKTEST RESULTS")
    print("=" * 80)
    print(f"Total Predictions: {metrics['total_predictions']}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print("=" * 80)
    
    # Save results
    results.to_csv(f"backtest_results_{args.symbol}_improved.csv", index=False)
    with open(f"backtest_metrics_{args.symbol}_improved.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n💾 Results saved to:")
    print(f"   backtest_results_{args.symbol}_improved.csv")
    
    print("\n" + "=" * 80)
    print("✅ IMPROVED TRAINING COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
