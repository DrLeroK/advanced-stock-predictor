"""
Advanced scaling with target encoding
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
import joblib

class RobustTimeSeriesScaler:
    """
    Robust scaler that handles outliers
    """
    def __init__(self):
        self.scaler = RobustScaler()
        self.fitted = False
    
    def fit(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.scaler.fit(X)
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        original_shape = X.shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        return X_scaled.reshape(original_shape)
    
    def inverse_transform(self, X):
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        original_shape = X.shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_inv = self.scaler.inverse_transform(X)
        return X_inv.reshape(original_shape)

class MultiTargetScaler:
    """
    Separate scalers for different targets
    """
    def __init__(self):
        self.return_scaler = RobustTimeSeriesScaler()
        self.volatility_scaler = RobustTimeSeriesScaler()
    
    def fit(self, returns, volatilities):
        self.return_scaler.fit(returns)
        self.volatility_scaler.fit(volatilities)
        return self
    
    def transform(self, returns, volatilities):
        returns_scaled = self.return_scaler.transform(returns)
        volatilities_scaled = self.volatility_scaler.transform(volatilities)
        return returns_scaled, volatilities_scaled