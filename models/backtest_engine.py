"""
Advanced Backtesting Engine with Comprehensive Metrics
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BacktestEngine:
    """
    Professional backtesting engine with walk-forward validation
    """
    
    def __init__(self, initial_capital=100000, transaction_cost=0.001, 
                 slippage=0.0005, risk_free_rate=0.02):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        
    def walk_forward_backtest(self, df, feature_cols, target_col, model_builder, 
                               sequence_length=60, train_years=3, test_days=20):
        """
        Walk-forward backtest that retrains model at each window
        """
        results = []
        capital = self.initial_capital
        positions = []
        trades = []
        
        # Create sequences
        X, y, dates = self._prepare_sequences(df, feature_cols, target_col, sequence_length)
        
        days_per_year = 252
        initial_train = train_years * days_per_year
        
        test_start = initial_train
        test_end = min(test_start + test_days, len(X))
        
        fold = 0
        
        while test_start < len(X):
            fold += 1
            print(f"   Fold {fold}: Training on {test_start} days, Testing {test_days} days")
            
            # Split data
            X_train = X[:test_start]
            y_train = y[:test_start]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            test_dates = dates[test_start:test_end]
            
            if len(X_test) == 0:
                break
            
            # Train model on this window
            model = model_builder()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Simulate trading
            for i, (pred, actual, date) in enumerate(zip(y_pred, y_test, test_dates)):
                trade_result = self._simulate_trade(capital, pred, actual)
                
                results.append({
                    'fold': fold,
                    'date': date,
                    'predicted_return': pred,
                    'actual_return': actual,
                    'direction_correct': (pred > 0) == (actual > 0),
                    'trade_signal': 'BUY' if pred > 0.005 else 'SELL' if pred < -0.005 else 'HOLD',
                    'capital': capital,
                    'pnl': trade_result['pnl']
                })
                
                if trade_result['trade_executed']:
                    trades.append(trade_result)
                capital = trade_result['new_capital']
            
            # Move window forward
            test_start += test_days
            test_end = min(test_start + test_days, len(X))
        
        # Calculate comprehensive metrics
        results_df = pd.DataFrame(results)
        metrics = self._calculate_metrics(results_df, trades)
        
        return results_df, metrics
    
    def _prepare_sequences(self, df, feature_cols, target_col, sequence_length):
        """Create sequences for time series"""
        X = df[feature_cols].values
        y = df[target_col].values
        dates = df['date'].values
        
        X_seq = []
        y_seq = []
        dates_seq = []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
            dates_seq.append(dates[i + sequence_length])
        
        return np.array(X_seq), np.array(y_seq), dates_seq
    
    def _simulate_trade(self, capital, predicted_return, actual_return):
        """Simulate a single trade with realistic costs"""
        trade_executed = False
        pnl = 0
        
        # Only trade when signal is strong (>0.5% expected)
        if abs(predicted_return) > 0.005:
            trade_executed = True
            position_size = capital * 0.95  # Use 95% of capital
            
            # Apply transaction costs and slippage
            cost = position_size * (self.transaction_cost + self.slippage)
            
            if predicted_return > 0:  # Buy
                pnl = position_size * actual_return - cost
            else:  # Sell short
                pnl = -position_size * actual_return - cost
            
            new_capital = capital + pnl
        else:
            new_capital = capital
        
        return {
            'trade_executed': trade_executed,
            'pnl': pnl,
            'new_capital': new_capital
        }
    
    def _calculate_metrics(self, results_df, trades):
        """Calculate comprehensive performance metrics"""
        
        # Directional metrics
        total_predictions = len(results_df)
        direction_accuracy = results_df['direction_correct'].mean()
        
        # Profit metrics
        if trades:
            total_trades = len(trades)
            win_rate = sum(1 for t in trades if t['pnl'] > 0) / total_trades if total_trades > 0 else 0
            avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in trades) else 0
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in trades) else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            total_trades = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Error metrics
        mae = np.mean(np.abs(results_df['predicted_return'] - results_df['actual_return']))
        rmse = np.sqrt(np.mean((results_df['predicted_return'] - results_df['actual_return']) ** 2))
        
        # Capital curve metrics
        capital_curve = results_df['capital'].values
        returns = np.diff(capital_curve) / capital_curve[:-1] if len(capital_curve) > 1 else [0]
        sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Drawdown
        cumulative = capital_curve / self.initial_capital
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio (return / max drawdown)
        total_return = (capital_curve[-1] - self.initial_capital) / self.initial_capital if len(capital_curve) > 0 else 0
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_predictions': total_predictions,
            'directional_accuracy': direction_accuracy,
            'mae': mae,
            'rmse': rmse,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'final_capital': capital_curve[-1] if len(capital_curve) > 0 else self.initial_capital
        }
