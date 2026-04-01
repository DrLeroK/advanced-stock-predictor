#!/usr/bin/env python
"""
Train standalone direction classifier model
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
from models.classifier import DirectionClassifier

def main():
    parser = argparse.ArgumentParser(description="Train direction classifier")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lstm-units", type=int, default=128)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎯 DIRECTION CLASSIFIER TRAINING")
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
    print(f"   📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
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
    
    # Create target (direction based on threshold)
    df['target_direction'] = (df['target_return'] > ModelConfig.UP_THRESHOLD).astype(int)
    
    # Drop NaN
    df_clean = df[['date'] + available_features + ['target_direction']].copy()
    df_clean = df_clean.dropna()
    
    print(f"   ✅ Cleaned data: {len(df_clean)} samples")
    
    # Scale features
    print("\n📐 Scaling features...")
    feature_scaler = RobustTimeSeriesScaler()
    X = df_clean[available_features].values.astype(np.float32)
    feature_scaler.fit(X)
    X_scaled = feature_scaler.transform(X)
    
    store.save_scaler(feature_scaler, args.symbol, "classifier_features")
    
    # Create sequences
    print("\n🔗 Creating sequences...")
    X_seq, y_seq = create_sequences(X_scaled, df_clean['target_direction'].values, args.sequence_length)
    
    print(f"   ✅ Created {len(X_seq)} sequences")
    print(f"   📊 Class balance: {np.sum(y_seq)} UP / {len(y_seq) - np.sum(y_seq)} DOWN")
    
    # Train/validation split
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
    
    # Build and train model
    print("\n🎓 Building classifier...")
    classifier = DirectionClassifier(
        sequence_length=args.sequence_length,
        n_features=len(available_features),
        lstm_units=args.lstm_units,
        dropout_rate=0.3,
        learning_rate=0.001
    )
    classifier.build()
    classifier.model.summary()
    
    print("\n🏋️ Training...")
    history = classifier.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    model_path = f"models/saved/{args.symbol}_classifier.keras"
    classifier.model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Evaluate
    print("\n📊 Final Evaluation:")
    y_pred_proba = classifier.predict_proba(X_val)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   AUC: {auc:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
