"""
Fundamental data integration
"""
import pandas as pd
import numpy as np

def add_fundamental_features(df, fundamentals_dict):
    """
    Add fundamental data as features
    """
    df = df.copy()
    
    if fundamentals_dict:
        # Add fundamental values as constant features
        for key, value in fundamentals_dict.items():
            if key != 'symbol' and key != 'date':
                df[f'fund_{key}'] = value
    
    return df

def create_earnings_feature(df, earnings_dates, days_before=3, days_after=3):
    """
    Add earnings period indicator
    """
    df = df.copy()
    df['is_earnings_period'] = 0
    df['days_since_earnings'] = 999
    
    for date in earnings_dates:
        start_date = date - pd.Timedelta(days=days_before)
        end_date = date + pd.Timedelta(days=days_after)
        
        mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
        df.loc[mask, 'is_earnings_period'] = 1
        
        # Days since earnings
        days_since = (df['date'].dt.date - date).dt.days
        df.loc[days_since > 0, 'days_since_earnings'] = days_since[days_since > 0]
    
    return df