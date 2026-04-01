"""
FastAPI application for advanced stock prediction
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import os
import sys
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas.requests import PredictionRequest, PredictionResponse, BacktestRequest
from models.classifier import DirectionClassifier
from models.regressor import HuberRegressor
from models.volatility_model import VolatilityPredictor
from models.ensemble import EnsemblePredictor
from data.store import AdvancedDataStore
from features.technical import add_advanced_technical_indicators
from features.volatility import add_volatility_features
from features.scalers import RobustTimeSeriesScaler
from config.settings import FEATURE_COLUMNS

app = FastAPI(
    title="Advanced Stock Prediction API",
    description="Ensemble LSTM model with direction, magnitude, and volatility prediction",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models cache
models_cache = {}

def get_models(symbol: str):
    """Load or get cached models for a symbol"""
    if symbol in models_cache:
        return models_cache[symbol]
    
    try:
        classifier = DirectionClassifier()
        classifier.model = tf.keras.models.load_model(f"models/saved/{symbol}_classifier.keras")
        
        regressor = HuberRegressor()
        regressor.model = tf.keras.models.load_model(f"models/saved/{symbol}_regressor.keras")
        
        volatility_model = VolatilityPredictor()
        volatility_model.model = tf.keras.models.load_model(f"models/saved/{symbol}_volatility.keras")
        
        store = AdvancedDataStore()
        feature_scaler = store.load_scaler(symbol, "features")
        
        ensemble = EnsemblePredictor()
        ensemble.fit(classifier, regressor, volatility_model)
        
        models_cache[symbol] = {
            'ensemble': ensemble,
            'feature_scaler': feature_scaler,
            'classifier': classifier,
            'regressor': regressor,
            'volatility': volatility_model
        }
        
        return models_cache[symbol]
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Models not found for {symbol}: {str(e)}")

@app.get("/")
async def root():
    return {
        "name": "Advanced Stock Prediction API",
        "version": "2.0.0",
        "description": "Ensemble LSTM model for stock prediction",
        "endpoints": [
            "/predict",
            "/health",
            "/models/{symbol}"
        ],
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cached_models": list(models_cache.keys())
    }

@app.get("/models/{symbol}")
async def get_model_info(symbol: str):
    """Get information about loaded models for a symbol"""
    models = get_models(symbol)
    return {
        "symbol": symbol,
        "loaded": True,
        "classifier": "loaded",
        "regressor": "loaded",
        "volatility": "loaded",
        "feature_count": len(models['feature_scaler'].scaler.feature_names_in_) if hasattr(models['feature_scaler'].scaler, 'feature_names_in_') else "unknown"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction for a stock"""
    models = get_models(request.symbol)
    
    # Fetch latest data
    from data.fetch import fetch_stock_data
    df = fetch_stock_data(request.symbol, period="1y")
    
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
    
    # Add features
    df = add_advanced_technical_indicators(df)
    df = add_volatility_features(df)
    
    # Get features
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    vol_features = ['realized_vol_20d', 'vol_ratio_20_60', 'parkinson_vol_20d']
    available_features.extend([f for f in vol_features if f in df.columns])
    
    # Prepare latest sequence
    df_clean = df[available_features].dropna()
    
    if len(df_clean) < 60:
        raise HTTPException(status_code=400, detail="Insufficient data for prediction")
    
    # Scale and create sequence
    X_scaled = models['feature_scaler'].transform(df_clean.values[-60:].reshape(1, -1))
    X_seq = X_scaled.reshape(1, 60, -1)
    
    # Make prediction
    result = models['ensemble'].predict_with_confidence(X_seq)
    
    # Get current price
    current_price = df['close'].iloc[-1]
    predicted_price = current_price * (1 + result['predicted_return'])
    
    return PredictionResponse(
        symbol=request.symbol,
        timestamp=datetime.now(),
        current_price=current_price,
        predicted_price=predicted_price,
        predicted_return=result['predicted_return'],
        predicted_return_pct=result['predicted_return_pct'],
        direction=result['direction'],
        direction_probability=result['direction_probability'],
        volatility=result['volatility'],
        confidence=result['confidence'],
        signal=result['signal'],
        components=result['components']
    )

@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """Run a backtest for a stock"""
    from models.backtest import AdvancedBacktester
    from scripts.backtest import run_backtest as run_backtest_script
    
    # This would run a full backtest
    # For brevity, we'll return a placeholder
    return {
        "symbol": request.symbol,
        "message": "Backtest functionality available via scripts/backtest.py",
        "status": "Use command line for full backtest"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
