#!/usr/bin/env python
"""
Universal Stock Prediction Script - Uses correct features for each stock
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

def get_expected_features(symbol):
    """
    Get the list of features that the model expects
    """
    feature_file = f"models/saved/{symbol}_features.txt"
    if os.path.exists(feature_file):
        with open(feature_file, 'r') as f:
            features = [line.strip() for line in f.readlines()]
        return features
    return None

def predict_stock(symbol):
    """
    Make prediction for a given stock symbol
    """
    print("=" * 80)
    print(f"🔮 ENSEMBLE PREDICTION - {symbol}")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Check if models exist
    model_files = {
        'classifier': f"models/saved/{symbol}_classifier.keras",
        'regressor': f"models/saved/{symbol}_regressor.keras",
        'volatility': f"models/saved/{symbol}_volatility.keras"
    }
    
    missing = [name for name, path in model_files.items() if not os.path.exists(path)]
    if missing:
        print(f"\n❌ Models not found for {symbol}! Missing: {', '.join(missing)}")
        print("   Available models:")
        import glob
        models = glob.glob("models/saved/*.keras")
        for m in models:
            print(f"      - {os.path.basename(m)}")
        print(f"\n💡 To train model for {symbol}:")
        print(f"   python scripts/train_ensemble.py --symbol {symbol}")
        return None
    
    # Get expected features
    expected_features = get_expected_features(symbol)
    if expected_features is None:
        print(f"\n❌ Feature list not found for {symbol}!")
        print(f"   Expected file: models/saved/{symbol}_features.txt")
        print(f"\n💡 Please retrain the model to save features:")
        print(f"   python scripts/train_ensemble.py --symbol {symbol}")
        return None
    
    print(f"\n📂 Loading models for {symbol}:")
    print(f"   📁 Classifier: {os.path.basename(model_files['classifier'])}")
    print(f"   📁 Regressor: {os.path.basename(model_files['regressor'])}")
    print(f"   📁 Volatility: {os.path.basename(model_files['volatility'])}")
    print(f"   📁 Features: {len(expected_features)} features expected")
    
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
            print("   ❌ No data from yfinance, trying cached...")
            store = AdvancedDataStore()
            df = store.load_raw(symbol)
            if df is None:
                print("   ❌ No cached data available")
                return None
        
        df = df.reset_index()
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        print(f"   ✅ Latest close: ${df['close'].iloc[-1]:.2f}")
        print(f"   📅 Latest date: {df['date'].iloc[-1].date()}")
        print(f"   📊 Total rows: {len(df)}")
    except Exception as e:
        print(f"   ❌ Error fetching data: {e}")
        return None
    
    # Add features
    print("\n📊 Adding features...")
    try:
        df = add_advanced_technical_indicators(df)
        df = add_volatility_features(df)
        
        # Get only the features the model expects
        available_features = [f for f in expected_features if f in df.columns]
        missing_features = set(expected_features) - set(available_features)
        
        if missing_features:
            print(f"   ⚠️ Missing {len(missing_features)} features: {list(missing_features)[:5]}...")
        
        # Select only expected features
        df_features = df[available_features].copy()
        df_features = df_features.dropna()
        
        # If we're missing features, pad with zeros
        if len(available_features) < len(expected_features):
            print(f"   ⚠️ Padding missing features with zeros")
            # Create a DataFrame with all expected features
            padded_df = pd.DataFrame(index=df_features.index)
            for f in expected_features:
                if f in df_features.columns:
                    padded_df[f] = df_features[f]
                else:
                    padded_df[f] = 0
            df_features = padded_df
        
        last_60 = df_features.iloc[-60:].values
        print(f"   📊 Using {last_60.shape[1]} features (expected {len(expected_features)})")
        
        # Scale
        X_scaled = scaler.transform(last_60)
        X_seq = X_scaled.reshape(1, 60, -1)
        
    except Exception as e:
        print(f"   ❌ Error processing features: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Make prediction
    print("\n🤖 Making prediction...")
    try:
        dir_proba = classifier.predict(X_seq, verbose=0)[0][0]
        ret_pred = regressor.predict(X_seq, verbose=0)[0][0]
        vol_pred = volatility_model.predict(X_seq, verbose=0)[0][0]
        
        # Fix volatility
        vol_pred = abs(vol_pred)
        vol_pred = min(vol_pred, 0.5)
        
        # Calculate confidence
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
        
    except Exception as e:
        print(f"   ❌ Error making prediction: {e}")
        return None
    
    # Display results
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
        print(f"   💡 Expected profit: {ret_pred*100:+.2f}% with {confidence:.0%} confidence")
    elif signal == "SELL":
        print("   📉 SELL - Consider short position or reduce exposure")
        print(f"   💡 Expected profit: {ret_pred*100:+.2f}% with {confidence:.0%} confidence")
    else:
        print("   ⏸️ HOLD - No strong signal, maintain current position")
    
    return {
        'symbol': symbol,
        'current_price': current_price,
        'predicted_price': predicted_price,
        'return_pct': ret_pred * 100,
        'direction': direction,
        'confidence': confidence,
        'signal': signal
    }

def main():
    parser = argparse.ArgumentParser(description="Predict stock prices")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol (AAPL, MSFT, GOOGL, etc.)")
    args = parser.parse_args()
    
    predict_stock(args.symbol.upper())

if __name__ == "__main__":
    main()
