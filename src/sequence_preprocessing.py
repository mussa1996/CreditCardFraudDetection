# src/sequence_preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_sequences(X, y, sequence_length=10):
    """
    Create sequences of transactions for LSTM input.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        sequence_length: Number of transactions in each sequence
    
    Returns:
        X_sequences: Sequence data (n_sequences, sequence_length, n_features)
        y_sequences: Target labels for sequences (n_sequences,)
    """
    X_sequences, y_sequences = [], []
    
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:(i + sequence_length)])
        y_sequences.append(y[i + sequence_length])  # Predict next transaction
    
    return np.array(X_sequences), np.array(y_sequences)

def prepare_sequential_data(df, sequence_length=10, test_size=0.2):
    """
    Prepare data for sequential modeling with temporal splitting.
    """
    # Sort by time to maintain temporal order
    if 'Time' in df.columns:
        df = df.sort_values('Time')
    
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    # Normalize features (important for LSTM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences
    X_sequences, y_sequences = create_sequences(X_scaled, y, sequence_length)
    
    # Temporal split (no shuffling to maintain time order)
    split_idx = int(len(X_sequences) * (1 - test_size))
    
    X_train = X_sequences[:split_idx]
    X_test = X_sequences[split_idx:]
    y_train = y_sequences[:split_idx]
    y_test = y_sequences[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler

def create_sliding_window_sequences(X, sequence_length=10):
    """
    Create sequences for real-time prediction using sliding window.
    """
    sequences = []
    for i in range(len(X) - sequence_length + 1):
        sequences.append(X[i:(i + sequence_length)])
    return np.array(sequences)