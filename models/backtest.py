"""
Advanced backtesting with proper directional accuracy calculation
"""
import numpy as np
import pandas as pd

class AdvancedBacktester:
    def __init__(self, initial_capital=10000, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
    
    def backtest(self, df, feature_cols, sequence_length, symbol, 
                 ensemble_model, feature_scaler, train_years=2, test_days=20):
        """
        Run walk-forward backtest with proper metrics
        """
        # Prepare sequences
        X = df[feature_cols].values
        X_scaled = feature_scaler.transform(X)
        
        # Create sequences
        X_seq = []
        y_seq = []
        dates_seq = []
        
        for i in range(len(X_scaled) - sequence_length):
            X_seq.append(X_scaled[i:i + sequence_length])
            y_seq.append(df['target_return'].iloc[i + sequence_length])
            dates_seq.append(df['date'].iloc[i + sequence_length])
        
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)
        
        days_per_year = 252
        initial_train_size = train_years * days_per_year
        
        results = []
        
        test_start = initial_train_size
        test_end = min(test_start + test_days, len(X_seq))
        
        fold = 0
        
        while test_start < len(X_seq):
            fold += 1
            print(f"   Fold {fold}: Test period {test_start} to {test_end}")
            
            X_test = X_seq[test_start:test_end]
            y_test = y_seq[test_start:test_end]
            test_dates = dates_seq[test_start:test_end]
            
            # Make predictions
            for i in range(len(X_test)):
                pred_result = ensemble_model.predict_with_confidence(X_test[i:i+1])
                predicted_return = pred_result['predicted_return']
                
                # Calculate direction correctness
                pred_direction = 1 if predicted_return > 0 else 0
                actual_direction = 1 if y_test[i] > 0 else 0
                direction_correct = 1 if pred_direction == actual_direction else 0
                
                # Store result
                results.append({
                    'fold': fold,
                    'date': test_dates[i],
                    'predicted_return': predicted_return,
                    'actual_return': y_test[i],
                    'direction_correct': direction_correct
                })
            
            test_start += test_days
            test_end = min(test_start + test_days, len(X_seq))
        
        # Calculate metrics
        results_df = pd.DataFrame(results)
        
        # Directional accuracy
        if len(results_df) > 0:
            direction_correct = results_df['direction_correct'].mean()
            mae = np.mean(np.abs(results_df['actual_return'] - results_df['predicted_return']))
            rmse = np.sqrt(np.mean((results_df['actual_return'] - results_df['predicted_return']) ** 2))
        else:
            direction_correct = 0
            mae = 0
            rmse = 0
        
        metrics = {
            'total_predictions': len(results_df),
            'directional_accuracy': direction_correct,
            'mae': mae,
            'rmse': rmse,
            'num_folds': fold
        }
        
        return results_df, metrics
