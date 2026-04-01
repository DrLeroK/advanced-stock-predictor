"""
Prediction routes
"""
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.schemas.requests import PredictionRequest, PredictionResponse
from api.app import get_models
from data.fetch import fetch_stock_data
from features.technical import add_advanced_technical_indicators
from features.volatility import add_volatility_features
from config.settings import FEATURE_COLUMNS

router = APIRouter(prefix="/predict", tags=["predictions"])

@router.post("/", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    """Make a prediction for a stock"""
    try:
        models = get_models(request.symbol)
        
        # Fetch latest data
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
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
