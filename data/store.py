"""
Enhanced data storage with fundamentals
"""
import pandas as pd
import os
import joblib
import json

class AdvancedDataStore:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "raw")
        self.processed_dir = os.path.join(base_dir, "processed")
        self.fundamentals_dir = os.path.join(base_dir, "fundamentals")
        self.scaler_dir = os.path.join(base_dir, "scalers")
        
        for dir_path in [self.raw_dir, self.processed_dir, 
                         self.fundamentals_dir, self.scaler_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def save_raw(self, df, symbol):
        filename = os.path.join(self.raw_dir, f"{symbol}.csv")
        df.to_csv(filename, index=False)
        return filename
    
    def load_raw(self, symbol):
        filename = os.path.join(self.raw_dir, f"{symbol}.csv")
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        return None
    
    def save_fundamentals(self, data, symbol):
        filename = os.path.join(self.fundamentals_dir, f"{symbol}.json")
        with open(filename, 'w') as f:
            json.dump(data, f, default=str)
        return filename
    
    def load_fundamentals(self, symbol):
        filename = os.path.join(self.fundamentals_dir, f"{symbol}.json")
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return None
    
    def save_processed(self, df, symbol, model_type):
        filename = os.path.join(self.processed_dir, f"{symbol}_{model_type}.csv")
        df.to_csv(filename, index=False)
        return filename
    
    def save_scaler(self, scaler, symbol, name):
        filename = os.path.join(self.scaler_dir, f"{symbol}_{name}.pkl")
        joblib.dump(scaler, filename)
        return filename
    
    def load_scaler(self, symbol, name):
        filename = os.path.join(self.scaler_dir, f"{symbol}_{name}.pkl")
        if os.path.exists(filename):
            return joblib.load(filename)
        return None