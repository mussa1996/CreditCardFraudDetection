# # # model_loader.py
# # import tensorflow as tf
# # import pandas as pd

# # # Path to the saved model
# # MODEL_PATH = "models/hybrid_model2.keras"

# # # Load the model
# # model = tf.keras.models.load_model(MODEL_PATH)

# # # Load feature columns from the dataset (excluding the target column)
# # df = pd.read_csv("data/raw/creditcard.csv")
# # FEATURE_COLUMNS = df.drop("Class", axis=1).columns.tolist()

# import os
# import tensorflow as tf
# import pandas as pd

# # Build absolute paths based on the file location
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "hybrid_model2.keras")
# DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")

# # Load the model
# model = tf.keras.models.load_model(MODEL_PATH)

# # Load feature columns
# df = pd.read_csv(DATA_PATH)
# FEATURE_COLUMNS = df.drop("Class", axis=1).columns.tolist()



# ============================================


# model_loader.py
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from src.config import SEQUENCE_LENGTH, FRAUD_THRESHOLD

# Build absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hybrid_lstm_model.keras")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Hybrid LSTM model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load feature columns
df = pd.read_csv(DATA_PATH)
FEATURE_COLUMNS = df.drop("Class", axis=1).columns.tolist()

# Load or create scaler
try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded successfully!")
except:
    print("❌ Scaler not found, will be created during training")
    scaler = None

class TransactionSequenceBuffer:
    """Buffer to maintain transaction sequences for real-time prediction"""
    
    def __init__(self, sequence_length=SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.buffer = []
        self.scaler = scaler
        
    def add_transaction(self, transaction_dict):
        """Add a new transaction to the buffer"""
        # Convert to DataFrame for consistent processing
        transaction_df = pd.DataFrame([transaction_dict])
        
        # Ensure correct feature order
        transaction_df = transaction_df[FEATURE_COLUMNS]
        
        # Scale the transaction
        if self.scaler:
            transaction_scaled = self.scaler.transform(transaction_df)
        else:
            transaction_scaled = transaction_df.values
            
        # Add to buffer
        self.buffer.append(transaction_scaled[0])
        
        # Maintain buffer size
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)
    
    def get_sequence(self):
        """Get current sequence for prediction"""
        if len(self.buffer) < self.sequence_length:
            # Pad with zeros if not enough transactions
            padding = np.zeros((self.sequence_length - len(self.buffer), len(FEATURE_COLUMNS)))
            sequence = np.vstack([padding, np.array(self.buffer)])
        else:
            sequence = np.array(self.buffer)
            
        return sequence.reshape(1, self.sequence_length, len(FEATURE_COLUMNS))
    
    def is_ready(self):
        """Check if buffer has enough transactions for prediction"""
        return len(self.buffer) >= self.sequence_length
    
    def clear(self):
        """Clear the buffer"""
        self.buffer = []

# Global buffer instance
prediction_buffer = TransactionSequenceBuffer()