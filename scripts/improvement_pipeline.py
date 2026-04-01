#!/usr/bin/env python
"""
Complete Improvement Pipeline - Run All Steps
"""
import sys
import os
import subprocess
import json
from datetime import datetime

print("=" * 80)
print("🚀 COMPLETE IMPROVEMENT PIPELINE")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Step 1: Hyperparameter Optimization
print("\n📊 Step 1: Hyperparameter Optimization")
print("-" * 50)
for symbol in ["AAPL", "MSFT", "GOOGL"]:
    print(f"\nOptimizing {symbol}...")
    subprocess.run(["python", "scripts/optimize_hyperparameters.py"])

# Step 2: Comprehensive Evaluation
print("\n📊 Step 2: Comprehensive Model Evaluation")
print("-" * 50)
for symbol in ["AAPL", "MSFT", "GOOGL"]:
    print(f"\nEvaluating {symbol}...")
    subprocess.run(["python", "scripts/comprehensive_evaluation.py"])

# Step 3: Compare Results
print("\n📊 Step 3: Comparing Results")
print("-" * 50)

print("\n📈 PERFORMANCE SUMMARY:")
print("-" * 80)
print(f"{'Stock':<10} {'Accuracy':<12} {'Sharpe':<10} {'Max DD':<10} {'Trades':<10}")
print("-" * 80)

for symbol in ["AAPL", "MSFT", "GOOGL"]:
    metrics_file = f"evaluation_metrics_{symbol}.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        print(f"{symbol:<10} {metrics['directional_accuracy']:.2%}    "
              f"{metrics['sharpe_ratio']:<10.2f} {metrics['max_drawdown']:<10.2%} "
              f"{metrics['total_trades']:<10}")

print("\n" + "=" * 80)
print("✅ IMPROVEMENT PIPELINE COMPLETE!")
print("=" * 80)
print("\n📁 Generated Files:")
print("   - optimization_results_*.json (hyperparameter tuning)")
print("   - evaluation_results_*.csv (backtest results)")
print("   - evaluation_metrics_*.json (performance metrics)")
print("   - evaluation_*.png (visualization charts)")
