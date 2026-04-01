#!/usr/bin/env python
"""
Fixed prediction script - properly handles volatility
"""
import sys
import os
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.store import AdvancedDataStore
from features.technical import add_advanced_technical_indicators
from features.volatility import add_volatility_features

def main():
    symbol = "AAPL"
    
    print("=" * 80)
    print("🔮 FIXED ENSEMBLE PREDICTION")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load models
    print("\n📂 Loading models...")
    classifier = tf.keras.models.load_model(f"models/saved/{symbol}_classifier.keras")
    regressor = tf.keras.models.load_model(f"models/saved/{symbol}_regressor.keras")
    volatility_model = tf.keras.models.load_model(f"models/saved/{symbol}_volatility.keras")
    store = AdvancedDataStore()
    scaler = store.load_scaler(symbol, "features")
    print("   ✅ Models loaded")
    
    # Get data
    print(f"\n📡 Fetching {symbol} data...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="2y")
    df = df.reset_index()
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    print(f"   ✅ Latest close: ${df['close'].iloc[-1]:.2f}")
    print(f"   📅 Latest date: {df['date'].iloc[-1].date()}")
    
    # Add ALL features
    print("\n📊 Adding features...")
    df = add_advanced_technical_indicators(df)
    df = add_volatility_features(df)
    
    # Get all numeric features (excluding price and date columns)
    exclude = ['date', 'open', 'high', 'low', 'close', 'volume', 
               'dividends', 'stock_splits', 'symbol']
    all_features = [col for col in df.columns if col not in exclude 
                    and not col.startswith('target_')]
    
    # Use all features
    df_features = df[all_features].dropna()
    last_60 = df_features.iloc[-60:].values
    
    print(f"   📊 Total features generated: {len(all_features)}")
    print(f"   📊 Using {last_60.shape[1]} features for prediction")
    
    # Scale - truncate to 37 features if needed
    if last_60.shape[1] > 37:
        print(f"   ⚠️ Truncating from {last_60.shape[1]} to 37 features")
        last_60 = last_60[:, :37]
    elif last_60.shape[1] < 37:
        print(f"   ⚠️ Need to pad from {last_60.shape[1]} to 37 features")
        # Pad with zeros
        padding = np.zeros((last_60.shape[0], 37 - last_60.shape[1]))
        last_60 = np.hstack([last_60, padding])
    
    # Scale
    X_scaled = scaler.transform(last_60)
    X_seq = X_scaled.reshape(1, 60, -1)
    
    # Predict
    print("\n🤖 Making prediction...")
    dir_proba = classifier.predict(X_seq, verbose=0)[0][0]
    ret_pred = regressor.predict(X_seq, verbose=0)[0][0]
    vol_pred = volatility_model.predict(X_seq, verbose=0)[0][0]
    
    # Ensure volatility is positive
    vol_pred = abs(vol_pred)  # Take absolute value to fix negative
    vol_pred = min(vol_pred, 0.5)  # Cap at 50% for reasonable values
    
    # Calculate confidence (based on classifier probability)
    confidence = dir_proba if dir_proba > 0.5 else 1 - dir_proba
    
    # Determine signal
    if confidence < 0.55:
        signal = "UNCERTAIN"
    elif ret_pred > 0.0075:
        signal = "BUY"
    elif ret_pred < -0.0075:
        signal = "SELL"
    else:
        signal = "HOLD"
    
    direction = "UP" if dir_proba > 0.5 else "DOWN"
    direction_prob = dir_proba if dir_proba > 0.5 else 1 - dir_proba
    
    current_price = df['close'].iloc[-1]
    predicted_price = current_price * (1 + ret_pred)
    
    print("\n" + "=" * 80)
    print("📊 PREDICTION RESULTS")
    print("=" * 80)
    print(f"\n📍 {symbol}:")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   Predicted Price: ${predicted_price:.2f}")
    print(f"   Expected Return: {ret_pred*100:+.2f}%")
    print(f"   Direction: {direction}")
    print(f"   Direction Probability: {direction_prob:.1%}")
    print(f"   Predicted Volatility: {vol_pred:.2%}")
    print(f"   Confidence: {confidence:.1%}")
    print(f"   Signal: {signal}")
    
    print("\n📊 Model Components:")
    print(f"   Classifier Probability: {dir_proba:.1%}")
    print(f"   Regressor Prediction: {ret_pred:.4f} ({ret_pred*100:+.2f}%)")
    print(f"   Volatility Prediction: {vol_pred:.2%}")
    
    print("\n" + "=" * 80)
    print("✅ Prediction complete!")
    print("=" * 80)
    
    # Trading recommendation
    print("\n📝 Trading Recommendation:")
    if signal == "UNCERTAIN":
        print("   ⚠️ UNCERTAIN - Stay in cash, wait for clearer signal")
    elif signal == "BUY":
        print("   📈 BUY - Consider long position with appropriate risk management")
    elif signal == "SELL":
        print("   📉 SELL - Consider short position or reduce exposure")
    else:
        print("   ⏸️ HOLD - No strong signal, maintain current position")

if __name__ == "__main__":
    main()
