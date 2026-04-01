"""
Backtest routes
"""
from fastapi import APIRouter, HTTPException
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.schemas.requests import BacktestRequest

router = APIRouter(prefix="/backtest", tags=["backtest"])

@router.post("/")
async def run_backtest(request: BacktestRequest):
    """Run a backtest for a stock"""
    try:
        # This would run a full backtest
        # For API, we return a reference to the script
        return {
            "symbol": request.symbol,
            "status": "Backtest functionality available via command line",
            "command": f"python scripts/backtest.py --symbol {request.symbol}",
            "initial_capital": request.initial_capital,
            "transaction_cost": request.transaction_cost
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{symbol}")
async def get_backtest_results(symbol: str):
    """Get saved backtest results"""
    import os
    import pandas as pd
    import json
    
    results_file = f"backtest_results_{symbol}.csv"
    metrics_file = f"backtest_metrics_{symbol}.json"
    
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        results = df.tail(100).to_dict(orient='records')
    else:
        results = None
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = None
    
    return {
        "symbol": symbol,
        "metrics": metrics,
        "recent_predictions": results,
        "file_exists": results is not None
    }
