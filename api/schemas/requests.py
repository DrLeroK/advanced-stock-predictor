"""
Pydantic schemas for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    confidence_threshold: Optional[float] = Field(0.5, ge=0, le=1, description="Minimum confidence for trade signal")

class PredictionResponse(BaseModel):
    symbol: str
    timestamp: datetime
    current_price: float
    predicted_price: float
    predicted_return: float
    predicted_return_pct: float
    direction: str
    direction_probability: float
    volatility: float
    confidence: float
    signal: str
    components: Dict[str, Any]

class BacktestRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    initial_capital: float = Field(10000, gt=0, description="Starting capital")
    transaction_cost: float = Field(0.001, ge=0, le=0.01, description="Transaction cost as percentage")
    train_years: int = Field(2, ge=1, le=5, description="Years for initial training")
    test_days: int = Field(20, ge=5, le=60, description="Days per test window")

class BacktestResponse(BaseModel):
    symbol: str
    metrics: Dict[str, Any]
    predictions: Optional[List[Dict[str, Any]]] = None

class ModelInfoResponse(BaseModel):
    symbol: str
    loaded: bool
    classifier: str
    regressor: str
    volatility: str
    feature_count: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    cached_models: List[str]
