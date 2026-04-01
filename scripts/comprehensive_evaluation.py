#!/usr/bin/env python
"""
Comprehensive Model Evaluation with Multiple Metrics
"""
import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.store import AdvancedDataStore
from features.technical import add_advanced_technical_indicators
from features.volatility import add_volatility_features
from features.feature_selector import FeatureSelector
from models.multi_ensemble import MultiAlgorithmEnsemble
from models.backtest_engine import BacktestEngine

def evaluate_model(symbol, use_ensemble=True):
    """
    Comprehensive evaluation of model performance
    """
    print("=" * 80)
    print(f"📊 COMPREHENSIVE EVALUATION - {symbol}")
    print("=" * 80)
    
    # Load and prepare data
    store = AdvancedDataStore()
    df = store.load_raw(symbol)
    
    if df is None:
        print(f"❌ No data for {symbol}")
        return None
    
    # Add features
    df = add_advanced_technical_indicators(df)
    df = add_volatility_features(df)
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in 
                    ['date', 'open', 'high', 'low', 'close', 'volume', 
                     'dividends', 'stock_splits', 'symbol', 'target_return', 
                     'target_volatility']]
    
    df_clean = df[['date'] + feature_cols + ['target_return']].dropna()
    
    print(f"\n📈 Data Statistics:")
    print(f"   Total samples: {len(df_clean)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
    
    # Feature selection
    print(f"\n🔍 Feature Selection...")
    X = df_clean[feature_cols].values
    y = df_clean['target_return'].values
    
    selector = FeatureSelector()
    selected_features, _ = selector.select_top_features(
        pd.DataFrame(X, columns=feature_cols), y, k=20
    )
    print(f"   Selected {len(selected_features)} features")
    
    # Create model builder
    def model_builder():
        if use_ensemble:
            return MultiAlgorithmEnsemble().build_ensemble()
        else:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=200, random_state=42)
    
    # Run backtest
    print(f"\n🔬 Running Walk-Forward Backtest...")
    backtest_engine = BacktestEngine(initial_capital=100000)
    
    results_df, metrics = backtest_engine.walk_forward_backtest(
        df_clean, selected_features, 'target_return', model_builder,
        sequence_length=60, train_years=3, test_days=20
    )
    
    # Print metrics
    print("\n" + "=" * 80)
    print("📊 EVALUATION METRICS")
    print("=" * 80)
    print(f"Total Predictions: {metrics['total_predictions']}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    print(f"MAE: {metrics['mae']:.4f} ({metrics['mae']*100:.2f}%)")
    print(f"RMSE: {metrics['rmse']:.4f} ({metrics['rmse']*100:.2f}%)")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Equity Curve
    axes[0, 0].plot(results_df['date'], results_df['capital'])
    axes[0, 0].set_title(f'{symbol} - Equity Curve')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Capital ($)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Predictions vs Actual
    axes[0, 1].scatter(results_df['actual_return'], results_df['predicted_return'], alpha=0.3)
    axes[0, 1].plot([-0.1, 0.1], [-0.1, 0.1], 'r--')
    axes[0, 1].set_title('Predictions vs Actual')
    axes[0, 1].set_xlabel('Actual Return')
    axes[0, 1].set_ylabel('Predicted Return')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Rolling Accuracy
    rolling_acc = results_df['direction_correct'].rolling(50).mean()
    axes[1, 0].plot(results_df['date'][50:], rolling_acc[50:])
    axes[1, 0].axhline(y=0.5, color='r', linestyle='--', label='Random')
    axes[1, 0].set_title('Rolling Directional Accuracy (50-day)')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Drawdown
    cumulative = results_df['capital'] / results_df['capital'].iloc[0]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    axes[1, 1].fill_between(results_df['date'], drawdown * 100, 0, alpha=0.5, color='red')
    axes[1, 1].set_title('Drawdown (%)')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Drawdown (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'evaluation_{symbol}.png', dpi=150)
    print(f"\n💾 Chart saved to evaluation_{symbol}.png")
    
    # Save results
    results_df.to_csv(f'evaluation_results_{symbol}.csv', index=False)
    with open(f'evaluation_metrics_{symbol}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"💾 Results saved to evaluation_results_{symbol}.csv")
    
    return results_df, metrics

if __name__ == "__main__":
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        evaluate_model(symbol, use_ensemble=True)
