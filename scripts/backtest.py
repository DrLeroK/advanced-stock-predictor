#!/usr/bin/env python
"""
Run backtest on trained ensemble model
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import ModelConfig, FEATURE_COLUMNS
from data.store import AdvancedDataStore
from features.technical import add_advanced_technical_indicators
from features.volatility import add_volatility_features
from features.fundamentals import add_fundamental_features
from features.scalers import RobustTimeSeriesScaler
from models.classifier import DirectionClassifier
from models.regressor import HuberRegressor
from models.volatility_model import VolatilityPredictor
from models.ensemble import EnsemblePredictor
from models.backtest import AdvancedBacktester

def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--initial-capital", type=float, default=10000)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔬 ADVANCED BACKTEST")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print(f"Transaction Cost: {args.transaction_cost*100:.1f}%")
    print("=" * 80)
    
    # Load models
    print("\n📂 Loading models...")
    
    try:
        classifier = DirectionClassifier()
        classifier.model = tf.keras.models.load_model(f"models/saved/{args.symbol}_classifier.keras")
        print("   ✅ Classifier loaded")
    except Exception as e:
        print(f"   ❌ Classifier not found: {e}")
        return
    
    try:
        regressor = HuberRegressor()
        regressor.model = tf.keras.models.load_model(f"models/saved/{args.symbol}_regressor.keras")
        print("   ✅ Regressor loaded")
    except Exception as e:
        print(f"   ❌ Regressor not found: {e}")
        return
    
    try:
        volatility_model = VolatilityPredictor()
        volatility_model.model = tf.keras.models.load_model(f"models/saved/{args.symbol}_volatility.keras")
        print("   ✅ Volatility model loaded")
    except Exception as e:
        print(f"   ❌ Volatility model not found: {e}")
        return
    
    store = AdvancedDataStore()
    feature_scaler = store.load_scaler(args.symbol, "features")
    if feature_scaler is None:
        print("   ❌ Feature scaler not found")
        return
    print("   ✅ Feature scaler loaded")
    
    ensemble = EnsemblePredictor()
    ensemble.fit(classifier, regressor, volatility_model)
    
    # Load data
    print("\n📂 Loading data for backtest...")
    df = store.load_raw(args.symbol)
    
    if df is None:
        print("❌ No data found")
        return
    
    print(f"   ✅ Loaded {len(df)} rows")
    
    # Add features
    print("\n📊 Adding features...")
    df = add_advanced_technical_indicators(df)
    df = add_volatility_features(df)
    
    # Get features
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    vol_features = ['realized_vol_20d', 'vol_ratio_20_60', 'parkinson_vol_20d']
    available_features.extend([f for f in vol_features if f in df.columns])
    
    fundamentals = store.load_fundamentals(args.symbol)
    if fundamentals:
        df = add_fundamental_features(df, fundamentals)
        fund_features = [col for col in df.columns if col.startswith('fund_')]
        available_features.extend(fund_features)
    
    # Prepare data for backtest
    df_clean = df[['date'] + available_features + ['target_return']].copy()
    df_clean = df_clean.dropna()
    
    print(f"   ✅ Prepared {len(df_clean)} samples")
    
    # Run backtest
    print("\n🔬 Running backtest...")
    backtester = AdvancedBacktester(
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost
    )
    
    results, metrics = backtester.backtest(
        df_clean, available_features, args.sequence_length,
        args.symbol, ensemble, feature_scaler,
        train_years=2, test_days=20
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 BACKTEST RESULTS")
    print("=" * 80)
    print(f"Total Predictions: {metrics['total_predictions']}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    print(f"MAE: {metrics['mae']:.4f} ({metrics['mae']*100:.2f}%)")
    print(f"RMSE: {metrics['rmse']:.4f} ({metrics['rmse']*100:.2f}%)")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Total Trades: {metrics['num_trades']}")
    print(f"Final Capital: ${metrics['final_capital']:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    
    # Save results
    results.to_csv(f"backtest_results_{args.symbol}.csv", index=False)
    with open(f"backtest_metrics_{args.symbol}.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n💾 Results saved to:")
    print(f"   backtest_results_{args.symbol}.csv")
    print(f"   backtest_metrics_{args.symbol}.json")
    
    print("\n" + "=" * 80)
    print("✅ BACKTEST COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
