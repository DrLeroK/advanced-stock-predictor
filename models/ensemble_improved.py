"""
Improved ensemble model with consistency checks
"""
import numpy as np

class ImprovedEnsemblePredictor:
    """
    Ensemble with consistency checking between classifier and regressor
    """
    
    def __init__(self, weights=None, consistency_threshold=0.6):
        self.classifier = None
        self.regressor = None
        self.volatility_model = None
        self.weights = weights or {'classifier': 0.4, 'regressor': 0.3, 'volatility': 0.3}
        self.consistency_threshold = consistency_threshold
        self.fitted = False
    
    def fit(self, classifier, regressor, volatility_model):
        self.classifier = classifier
        self.regressor = regressor
        self.volatility_model = volatility_model
        self.fitted = True
        return self
    
    def predict_with_confidence(self, X):
        """
        Make prediction with consistency checking
        """
        if not self.fitted:
            raise ValueError("Ensemble not fitted")
        
        # Get predictions
        dir_proba = self.classifier.predict_proba(X).flatten()[0]
        magnitude = self.regressor.predict(X).flatten()[0]
        volatility = self.volatility_model.predict(X).flatten()[0]
        
        # Determine direction
        direction = "UP" if dir_proba > 0.5 else "DOWN"
        direction_prob = dir_proba if dir_proba > 0.5 else 1 - dir_proba
        
        # CHECK CONSISTENCY
        # If classifier says DOWN but regressor predicts positive return
        if direction == "DOWN" and magnitude > 0:
            # Models disagree - reduce confidence significantly
            consistency_penalty = 0.5
            print(f"   ⚠️ Consistency check: DOWN direction but positive magnitude")
        # If classifier says UP but regressor predicts negative return
        elif direction == "UP" and magnitude < 0:
            consistency_penalty = 0.5
            print(f"   ⚠️ Consistency check: UP direction but negative magnitude")
        else:
            consistency_penalty = 1.0
        
        # Apply consistency penalty
        adjusted_confidence = direction_prob * consistency_penalty
        
        # Adjust magnitude based on confidence
        if adjusted_confidence < self.consistency_threshold:
            # Low confidence - reduce magnitude
            adjusted_magnitude = magnitude * 0.3
            signal = "UNCERTAIN"
        else:
            adjusted_magnitude = magnitude
            # Determine signal
            if adjusted_magnitude > 0.0075:
                signal = "BUY"
            elif adjusted_magnitude < -0.0075:
                signal = "SELL"
            else:
                signal = "HOLD"
        
        # Fix volatility (positive and reasonable)
        volatility = abs(volatility)
        volatility = min(volatility, 0.5)
        
        # Final confidence is the adjusted probability
        confidence = adjusted_confidence
        
        return {
            'predicted_return': float(adjusted_magnitude),
            'predicted_return_pct': float(adjusted_magnitude * 100),
            'direction': direction,
            'direction_probability': float(direction_prob),
            'volatility': float(volatility),
            'confidence': float(confidence),
            'signal': signal,
            'components': {
                'classifier': float(dir_proba),
                'regressor': float(magnitude),
                'volatility': float(volatility),
                'consistency_penalty': consistency_penalty
            }
        }
    
    def predict_batch(self, X):
        results = []
        for i in range(len(X)):
            result = self.predict_with_confidence(X[i:i+1])
            results.append(result)
        return results
