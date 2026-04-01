"""
Multi-Algorithm Ensemble Model
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

class MultiAlgorithmEnsemble:
    """
    Ensemble of multiple ML algorithms with meta-learner
    """
    
    def __init__(self):
        self.base_models = []
        self.meta_model = None
        self.fitted = False
        
    def build_ensemble(self):
        """Build the ensemble of diverse algorithms"""
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        
        # Support Vector Machine (with calibration)
        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='auto',
            probability=True,
            random_state=42
        )
        svm_calibrated = CalibratedClassifierCV(svm, method='platt', cv=3)
        
        # Logistic Regression
        lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )
        
        self.base_models = [
            ('RandomForest', rf),
            ('GradientBoosting', gb),
            ('SVM', svm_calibrated),
            ('LogisticRegression', lr)
        ]
        
        # Meta-learner
        self.meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        
        return self
    
    def fit(self, X, y):
        """Fit all base models and meta-learner"""
        if not self.base_models:
            self.build_ensemble()
        
        # Get predictions from base models using cross-validation
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for i, (name, model) in enumerate(self.base_models):
            print(f"   Training {name}...")
            for train_idx, val_idx in skf.split(X, y):
                model.fit(X[train_idx], y[train_idx])
                meta_features[val_idx, i] = model.predict_proba(X[val_idx])[:, 1]
        
        # Train meta-learner
        self.meta_model.fit(meta_features, y)
        self.fitted = True
        
        # Refit base models on all data
        for name, model in self.base_models:
            model.fit(X, y)
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities using ensemble"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        # Get base model predictions
        meta_features = np.zeros((len(X), len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models):
            meta_features[:, i] = model.predict_proba(X)[:, 1]
        
        # Meta-learner prediction
        return self.meta_model.predict_proba(meta_features)
    
    def predict(self, X):
        """Predict classes"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def get_feature_importance(self):
        """Get feature importance (if applicable)"""
        importance = {}
        for name, model in self.base_models:
            if hasattr(model, 'feature_importances_'):
                importance[name] = model.feature_importances_
        return importance
