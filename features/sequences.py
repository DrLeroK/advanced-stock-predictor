"""
Sequence creation with multiple horizons
"""
import numpy as np

def create_sequences(X, y, sequence_length=60):
    """
    Create sequences for LSTM input
    """
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

def create_multihorizon_sequences(X, y_dict, sequence_length=60):
    """
    Create sequences for multiple prediction horizons
    """
    X_seq = []
    y_seqs = {horizon: [] for horizon in y_dict.keys()}
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        for horizon, y in y_dict.items():
            y_seqs[horizon].append(y[i + sequence_length])
    
    X_seq = np.array(X_seq, dtype=np.float32)
    for horizon in y_seqs:
        y_seqs[horizon] = np.array(y_seqs[horizon], dtype=np.float32)
    
    return X_seq, y_seqs

def create_volatility_sequences(X, y_returns, y_volatility, sequence_length=60):
    """
    Create sequences for volatility-conditioned return prediction
    """
    X_seq = []
    y_returns_seq = []
    y_volatility_seq = []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_returns_seq.append(y_returns[i + sequence_length])
        y_volatility_seq.append(y_volatility[i + sequence_length])
    
    return (np.array(X_seq, dtype=np.float32),
            np.array(y_returns_seq, dtype=np.float32),
            np.array(y_volatility_seq, dtype=np.float32))