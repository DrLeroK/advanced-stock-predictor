"""
Ensemble model combining classifier, regressor, and volatility
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

class EnsemblePredictor:
    """
    Ensemble that combines multiple models for robust predictions
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        self.classifier = None
        self.regressor = None
        self.volatility_model = None
        self.weights = weights or {'classifier': 0.4, 'regressor': 0.3, 'volatility': 0.3}
        self.fitted = False
    
    def fit(self, classifier, regressor, volatility_model):
        """Set the component models"""
        self.classifier = classifier
        self.regressor = regressor
        self.volatility_model = volatility_model
        self.fitted = True
        return self
    
    def predict_with_confidence(self, X):
        """
        Make ensemble prediction with confidence intervals
        """
        if not self.fitted:
            raise ValueError("Ensemble not fitted")
        
        # Get predictions from each model
        direction_proba = self.classifier.predict_proba(X).flatten()
        direction = (direction_proba > 0.5).astype(int)
        
        magnitude = self.regressor.predict(X)
        volatility = self.volatility_model.predict(X)
        
        # Conditional prediction based on volatility regime
        # In high volatility, reduce position size (represented by lower magnitude)
        volatility_scale = np.clip(1.0 / (1.0 + volatility), 0.5, 1.0)
        
        # Combine predictions
        # For upward predictions: positive magnitude, for downward: negative magnitude
        base_prediction = magnitude * (2 * direction - 1)
        
        # Adjust by volatility
        adjusted_prediction = base_prediction * volatility_scale
        
        # Confidence based on classifier probability and volatility
        confidence = direction_proba * (1.0 - np.clip(volatility, 0, 0.5))
        
        # Get signal
        signal = self._get_signal(adjusted_prediction[0], confidence[0])
        
        return {
            'predicted_return': float(adjusted_prediction[0]),
            'predicted_return_pct': float(adjusted_prediction[0] * 100),
            'direction': 'UP' if direction[0] == 1 else 'DOWN',
            'direction_probability': float(direction_proba[0]),
            'volatility': float(volatility[0]),
            'confidence': float(confidence[0]),
            'signal': signal,
            'components': {
                'classifier': float(direction_proba[0]),
                'regressor': float(magnitude[0]),
                'volatility': float(volatility[0])
            }
        }
    
    def _get_signal(self, predicted_return, confidence):
        """Convert prediction to trading signal"""
        if confidence < 0.5:
            return "UNCERTAIN"
        
        if predicted_return > 0.015:
            return "STRONG_BUY"
        elif predicted_return > 0.0075:
            return "BUY"
        elif predicted_return < -0.015:
            return "STRONG_SELL"
        elif predicted_return < -0.0075:
            return "SELL"
        else:
            return "HOLD"
    
    def predict_batch(self, X):
        """Batch prediction for multiple samples"""
        results = []
        for i in range(len(X)):
            result = self.predict_with_confidence(X[i:i+1])
            results.append(result)
        return results