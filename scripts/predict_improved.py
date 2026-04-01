#!/usr/bin/env python
"""
Improved prediction script with consistency checking
"""
import sys
import os
import argparse
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.store import AdvancedDataStore
from features.technical import add_advanced_technical_indicators
from features.volatility import add_volatility_features
from models.ensemble_improved import ImprovedEnsemblePredictor

def predict_stock_improved(symbol):
    """
    Make improved prediction with consistency checking
    """
    print("=" * 80)
    print(f"🔮 IMPROVED ENSEMBLE PREDICTION - {symbol}")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Check feature file
    feature_file = f"models/saved/{symbol}_features.txt"
    selected_file = f"models/saved/{symbol}_selected_features.txt"
    
    if os.path.exists(selected_file):
        feature_file = selected_file
        print(f"\n📋 Using selected features: {selected_file}")
    elif os.path.exists(feature_file):
        print(f"\n📋 Using full features: {feature_file}")
    else:
        print(f"\n❌ Feature list not found for {symbol}!")
        return None
    
    with open(feature_file, 'r') as f:
        expected_features = [line.strip() for line in f.readlines()]
    
    print(f"   📊 Expecting {len(expected_features)} features")
    
    # Load models
    model_files = {
        'classifier': f"models/saved/{symbol}_classifier.keras",
        'regressor': f"models/saved/{symbol}_regressor.keras",
        'volatility': f"models/saved/{symbol}_volatility.keras"
    }
    
    print(f"\n📂 Loading models for {symbol}:")
    for name, path in model_files.items():
        print(f"   📁 {name.capitalize()}: {os.path.basename(path)}")
    
    try:
        classifier = tf.keras.models.load_model(model_files['classifier'])
        regressor = tf.keras.models.load_model(model_files['regressor'])
        volatility_model = tf.keras.models.load_model(model_files['volatility'])
        store = AdvancedDataStore()
        scaler = store.load_scaler(symbol, "features")
        print("   ✅ Models loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading models: {e}")
        return None
    
    # Fetch data
    print(f"\n📡 Fetching {symbol} data...")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="2y")
        
        if df.empty:
            df = store.load_raw(symbol)
            if df is None:
                print("   ❌ No data available")
                return None
        
        df = df.reset_index()
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        print(f"   ✅ Latest close: ${df['close'].iloc[-1]:.2f}")
        print(f"   📅 Latest date: {df['date'].iloc[-1].date()}")
    except Exception as e:
        print(f"   ❌ Error fetching data: {e}")
        return None
    
    # Add features
    print("\n📊 Adding features...")
    try:
        df = add_advanced_technical_indicators(df)
        df = add_volatility_features(df)
        
        # Get expected features
        available_features = [f for f in expected_features if f in df.columns]
        
        if len(available_features) < len(expected_features):
            missing = set(expected_features) - set(available_features)
            print(f"   ⚠️ Missing {len(missing)} features")
            # Pad with zeros for missing features
            for f in missing:
                df[f] = 0
        
        # Select features in correct order
        df_features = df[expected_features].copy()
        df_features = df_features.dropna()
        
        last_60 = df_features.iloc[-60:].values
        print(f"   📊 Using {last_60.shape[1]} features")
        
        # Scale
        X_scaled = scaler.transform(last_60)
        X_seq = X_scaled.reshape(1, 60, -1)
        
    except Exception as e:
        print(f"   ❌ Error processing features: {e}")
        return None
    
    # Make prediction with improved ensemble
    print("\n🤖 Making prediction with consistency check...")
    
    try:
        # Get individual predictions
        dir_proba = classifier.predict(X_seq, verbose=0)[0][0]
        magnitude = regressor.predict(X_seq, verbose=0)[0][0]
        volatility = volatility_model.predict(X_seq, verbose=0)[0][0]
        
        # Use improved ensemble logic
        direction = "UP" if dir_proba > 0.5 else "DOWN"
        direction_prob = dir_proba if dir_proba > 0.5 else 1 - dir_proba
        
        # Consistency check
        consistency_penalty = 1.0
        if direction == "DOWN" and magnitude > 0:
            consistency_penalty = 0.5
            print(f"   ⚠️ Consistency warning: DOWN direction but positive magnitude")
        elif direction == "UP" and magnitude < 0:
            consistency_penalty = 0.5
            print(f"   ⚠️ Consistency warning: UP direction but negative magnitude")
        
        adjusted_confidence = direction_prob * consistency_penalty
        adjusted_magnitude = magnitude
        
        if adjusted_confidence < 0.55:
            adjusted_magnitude = magnitude * 0.3
            signal = "UNCERTAIN"
        elif adjusted_magnitude > 0.0075:
            signal = "BUY"
        elif adjusted_magnitude < -0.0075:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        # Fix volatility
        volatility = abs(volatility)
        volatility = min(volatility, 0.5)
        
        current_price = df['close'].iloc[-1]
        predicted_price = current_price * (1 + adjusted_magnitude)
        
    except Exception as e:
        print(f"   ❌ Error making prediction: {e}")
        return None
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 IMPROVED PREDICTION RESULTS")
    print("=" * 80)
    print(f"\n📍 {symbol}:")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   Predicted Price: ${predicted_price:.2f}")
    print(f"   Expected Return: {adjusted_magnitude*100:+.2f}%")
    print(f"   Direction: {direction}")
    print(f"   Direction Probability: {direction_prob:.1%}")
    print(f"   Consistency Penalty: {consistency_penalty:.0%}")
    print(f"   Adjusted Confidence: {adjusted_confidence:.1%}")
    print(f"   Predicted Volatility: {volatility:.2%}")
    print(f"   Signal: {signal}")
    
    print("\n📊 Model Components:")
    print(f"   Classifier: {dir_proba:.1%}")
    print(f"   Regressor: {magnitude:.4f} ({magnitude*100:+.2f}%)")
    print(f"   Volatility: {volatility:.2%}")
    
    print("\n" + "=" * 80)
    print("✅ Prediction complete!")
    print("=" * 80)
    
    # Trading recommendation with position sizing
    print("\n📝 Trading Recommendation:")
    if signal == "UNCERTAIN":
        print("   ⚠️ UNCERTAIN - Models disagree or low confidence")
        print("   💡 Recommended: Stay in cash")
    elif signal == "BUY":
        position_size = min(0.95 * (adjusted_confidence / 0.5), 0.95)
        print(f"   📈 BUY - Consider LONG position")
        print(f"   💡 Position Size: {position_size:.0%} of capital")
        print(f"   💡 Expected Return: {adjusted_magnitude*100:+.2f}%")
        print(f"   💡 Risk: {volatility:.2%} volatility")
    elif signal == "SELL":
        position_size = min(0.95 * (adjusted_confidence / 0.5), 0.95)
        print(f"   📉 SELL - Consider SHORT position")
        print(f"   💡 Position Size: {position_size:.0%} of capital")
        print(f"   💡 Expected Return: {adjusted_magnitude*100:+.2f}%")
        print(f"   💡 Risk: {volatility:.2%} volatility")
    else:
        print("   ⏸️ HOLD - No clear signal")
        print("   💡 Recommended: Maintain current position")
    
    return {
        'symbol': symbol,
        'current_price': current_price,
        'predicted_price': predicted_price,
        'return_pct': adjusted_magnitude * 100,
        'direction': direction,
        'confidence': adjusted_confidence,
        'signal': signal,
        'volatility': volatility
    }

def main():
    parser = argparse.ArgumentParser(description="Improved stock prediction")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol")
    args = parser.parse_args()
    
    predict_stock_improved(args.symbol.upper())

if __name__ == "__main__":
    main()
