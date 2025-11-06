# # src/config.py
# import os

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # Data paths
# RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'creditcard.csv')
# PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_creditcard.csv')

# # Model parameters
# # BATCH_SIZE = 64
# # EPOCHS = 50
# # LEARNING_RATE = 0.001
# BATCH_SIZE = 32      # reduced batch size in order to have a better convergence
# EPOCHS = 100           # increased epochs in order to have a better convergence
# LEARNING_RATE = 0.0005 # lowered learning rate inorder to have a better convergence
# TEST_SIZE = 0.2
# RANDOM_STATE = 42


# # Model save path
# MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'hybrid_model2.keras')



# =================================================


# src/config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'creditcard.csv')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_creditcard.csv')

# Model parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0005
TEST_SIZE = 0.2
RANDOM_STATE = 42
SEQUENCE_LENGTH = 10  # Number of transactions in each sequence

# Model save path
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'hybrid_lstm_model.keras')

# Evaluation thresholds
FRAUD_THRESHOLD = 0.5