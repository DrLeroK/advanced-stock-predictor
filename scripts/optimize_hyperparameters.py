"""
Hyperparameter Optimization with Grid Search and Cross-Validation
"""
import sys
import os
import itertools
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.store import AdvancedDataStore
from features.technical import add_advanced_technical_indicators
from features.volatility import add_volatility_features
from features.feature_selector import FeatureSelector
from models.classifier import DirectionClassifier
from models.regressor import HuberRegressor
from models.backtest_engine import BacktestEngine

class HyperparameterOptimizer:
    """
    Grid search for optimal model hyperparameters
    """
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.store = AdvancedDataStore()
        self.best_params = {}
        self.best_score = -np.inf
        
    def load_and_prepare_data(self):
        """Load and prepare data for optimization"""
        df = self.store.load_raw(self.symbol)
        if df is None:
            return None
        
        # Add features
        df = add_advanced_technical_indicators(df)
        df = add_volatility_features(df)
        
        # Create target
        df['target_direction'] = (df['target_return'] > 0.005).astype(int)
        
        # Clean data
        feature_cols = [col for col in df.columns if col not in 
                        ['date', 'open', 'high', 'low', 'close', 'volume', 
                         'dividends', 'stock_splits', 'symbol', 'target_return', 
                         'target_direction', 'target_volatility']]
        
        df_clean = df[['date'] + feature_cols + ['target_direction', 'target_return']].dropna()
        
        return df_clean, feature_cols
    
    def optimize_classifier(self, df_clean, feature_cols):
        """Optimize classifier hyperparameters"""
        print(f"\n🎯 Optimizing Classifier for {self.symbol}")
        print("-" * 50)
        
        # Define hyperparameter grid
        param_grid = {
            'lstm_units': [32, 64, 128],
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.0005],
            'sequence_length': [30, 60, 90]
        }
        
        best_score = 0
        best_params = {}
        
        for lstm_units in param_grid['lstm_units']:
            for dropout in param_grid['dropout_rate']:
                for lr in param_grid['learning_rate']:
                    for seq_len in param_grid['sequence_length']:
                        
                        print(f"   Testing: LSTM={lstm_units}, Dropout={dropout}, LR={lr}, Seq={seq_len}")
                        
                        try:
                            # Create model
                            model = DirectionClassifier(
                                sequence_length=seq_len,
                                n_features=len(feature_cols),
                                lstm_units=lstm_units,
                                dropout_rate=dropout,
                                learning_rate=lr
                            )
                            model.build()
                            
                            # Simple validation (using last 20% for validation)
                            X = df_clean[feature_cols].values
                            y = df_clean['target_direction'].values
                            
                            # Create sequences
                            from features.sequences import create_sequences
                            X_seq, y_seq = create_sequences(X, y, seq_len)
                            
                            # Split
                            split = int(len(X_seq) * 0.8)
                            X_train, X_val = X_seq[:split], X_seq[split:]
                            y_train, y_val = y_seq[:split], y_seq[split:]
                            
                            # Train briefly
                            history = model.train(X_train, y_train, X_val, y_val, epochs=10)
                            
                            # Get validation accuracy
                            val_acc = history.history['val_accuracy'][-1]
                            
                            if val_acc > best_score:
                                best_score = val_acc
                                best_params = {
                                    'lstm_units': lstm_units,
                                    'dropout_rate': dropout,
                                    'learning_rate': lr,
                                    'sequence_length': seq_len
                                }
                                print(f"      ✅ New best: {val_acc:.2%}")
                                
                        except Exception as e:
                            print(f"      ❌ Error: {e}")
        
        print(f"\n🏆 Best Classifier Params: {best_params} (Acc: {best_score:.2%})")
        return best_params, best_score
    
    def run_full_optimization(self):
        """Run complete hyperparameter optimization"""
        print("=" * 80)
        print(f"🔧 HYPERPARAMETER OPTIMIZATION - {self.symbol}")
        print("=" * 80)
        
        df_clean, feature_cols = self.load_and_prepare_data()
        if df_clean is None:
            return
        
        # Optimize classifier
        best_classifier, best_acc = self.optimize_classifier(df_clean, feature_cols)
        
        # Save results
        results = {
            'symbol': self.symbol,
            'best_classifier_params': best_classifier,
            'best_accuracy': best_acc,
            'feature_count': len(feature_cols),
            'optimization_date': datetime.now().isoformat()
        }
        
        import json
        with open(f"optimization_results_{self.symbol}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to optimization_results_{self.symbol}.json")
        
        return results

if __name__ == "__main__":
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        optimizer = HyperparameterOptimizer(symbol)
        optimizer.run_full_optimization()
