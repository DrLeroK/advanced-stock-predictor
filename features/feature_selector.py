"""
Advanced feature selection for stock prediction
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.decomposition import PCA

class FeatureSelector:
    """
    Intelligent feature selection to reduce overfitting
    """
    
    @staticmethod
    def select_top_features(X, y, k=15):
        """
        Select top K features using mutual information
        """
        selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
        selector.fit(X, y)
        
        # Get feature scores
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        selected_features = scores.head(k)['feature'].tolist()
        
        print(f"   ✅ Selected top {k} features")
        print(f"   📊 Top 5 features: {selected_features[:5]}")
        
        return selected_features, selector
    
    @staticmethod
    def reduce_dimensions_pca(X, n_components=20):
        """
        Use PCA for dimensionality reduction
        """
        pca = PCA(n_components=min(n_components, X.shape[1]))
        X_reduced = pca.fit_transform(X)
        explained_variance = pca.explained_variance_ratio_.sum()
        
        print(f"   ✅ PCA reduced from {X.shape[1]} to {n_components} dimensions")
        print(f"   📊 Explained variance: {explained_variance:.2%}")
        
        return X_reduced, pca
    
    @staticmethod
    def remove_correlated_features(X, threshold=0.95):
        """
        Remove highly correlated features
        """
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        if to_drop:
            print(f"   ✅ Removing {len(to_drop)} highly correlated features")
            return X.drop(columns=to_drop), to_drop
        
        return X, []
