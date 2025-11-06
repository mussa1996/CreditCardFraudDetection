# # src/model.py
# import tensorflow as tf
# from src.config import LEARNING_RATE

# Model = tf.keras.Model
# Input = tf.keras.Input
# Dense = tf.keras.layers.Dense
# Adam = tf.keras.optimizers.Adam

# def create_hybrid_model(input_dim):
#     """
#     Create a hybrid model with two outputs:
#     - A classification branch (using binary crossentropy loss)
#     - A reconstruction branch (using MSE loss for autoencoder reconstruction)
#     """
#     # Input layer
#     inputs = Input(shape=(input_dim,))
    
#     # Shared encoding layers
#     x = Dense(64, activation='relu')(inputs)
#     x = Dense(32, activation='relu')(x)
#     encoded = Dense(16, activation='relu')(x)
    
#     # Classification branch
#     classification_output = Dense(1, activation='sigmoid', name='classification')(encoded)
    
#     # Reconstruction branch (decoder)
#     reconstruction = Dense(32, activation='relu')(encoded)
#     reconstruction = Dense(64, activation='relu')(reconstruction)
#     reconstruction_output = Dense(input_dim, activation='linear', name='reconstruction')(reconstruction)
    
#     # Define multi-output model
#     model = Model(inputs=inputs, outputs=[classification_output, reconstruction_output])
    
#     # Compile the model with different loss weights for each output
#     model.compile(
#         optimizer=Adam(learning_rate=LEARNING_RATE),
#         loss={'classification': 'binary_crossentropy', 'reconstruction': 'mse'},
#         loss_weights={'classification': 1.0, 'reconstruction': 0.5},
#         metrics={'classification': 'accuracy'}
#     )
    
#     return model



# =============================================

# src/model.py
import tensorflow as tf
from src.config import LEARNING_RATE

Model = tf.keras.Model
Input = tf.keras.Input
Dense = tf.keras.layers.Dense
LSTM = tf.keras.layers.LSTM
Bidirectional = tf.keras.layers.Bidirectional
Dropout = tf.keras.layers.Dropout
Adam = tf.keras.optimizers.Adam

def create_hybrid_model(input_dim, sequence_length=10):
    """
    Create a hybrid LSTM + Autoencoder model for fraud detection.
    
    Args:
        input_dim: Number of features per transaction
        sequence_length: Number of transactions in each sequence
    
    Returns:
        model: Compiled hybrid model
    """
    # Input layer for sequences
    sequence_input = Input(shape=(sequence_length, input_dim), name='sequence_input')
    
    # LSTM branch for sequential pattern recognition
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(sequence_input)
    lstm_layer = LSTM(32, dropout=0.2)(lstm_layer)
    
    # Autoencoder branch
    # Flatten the sequence for autoencoder (or use last output)
    encoded = Dense(64, activation='relu', name='encoder_dense1')(lstm_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(32, activation='relu', name='encoder_dense2')(encoded)
    bottleneck = Dense(16, activation='relu', name='bottleneck')(encoded)
    
    # Classification branch
    classification_output = Dense(1, activation='sigmoid', name='classification')(bottleneck)
    
    # Reconstruction branch (decoder)
    decoder = Dense(32, activation='relu', name='decoder_dense1')(bottleneck)
    decoder = Dense(64, activation='relu', name='decoder_dense2')(decoder)
    decoder = Dense(input_dim, activation='linear', name='decoder_dense3')(decoder)
    
    # For reconstruction, we need to match the sequence length
    # We'll reconstruct the last transaction in the sequence
    reconstruction_output = Dense(input_dim, activation='linear', name='reconstruction')(decoder)
    
    # Define the model
    model = Model(
        inputs=sequence_input, 
        outputs=[classification_output, reconstruction_output]
    )
    
    # Compile with different loss weights
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss={
            'classification': 'binary_crossentropy', 
            'reconstruction': 'mse'
        },
        loss_weights={
            'classification': 1.0, 
            'reconstruction': 0.3  # Lower weight for reconstruction
        },
        metrics={
            'classification': ['accuracy', 'precision', 'recall'],
            'reconstruction': ['mse']
        }
    )
    
    print("Hybrid LSTM + Autoencoder Model Summary:")
    model.summary()
    
    return model

def create_simple_hybrid_model(input_dim, sequence_length=10):
    """
    Simplified version for faster training.
    """
    sequence_input = Input(shape=(sequence_length, input_dim), name='sequence_input')
    
    # LSTM layers
    lstm_out = LSTM(32, dropout=0.2)(sequence_input)
    
    # Autoencoder
    encoded = Dense(16, activation='relu')(lstm_out)
    bottleneck = Dense(8, activation='relu')(encoded)
    
    # Classification
    classification_output = Dense(1, activation='sigmoid', name='classification')(bottleneck)
    
    # Reconstruction
    decoder = Dense(16, activation='relu')(bottleneck)
    reconstruction_output = Dense(input_dim, activation='linear', name='reconstruction')(decoder)
    
    model = Model(
        inputs=sequence_input, 
        outputs=[classification_output, reconstruction_output]
    )
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss={
            'classification': 'binary_crossentropy', 
            'reconstruction': 'mse'
        },
        loss_weights={
            'classification': 1.0, 
            'reconstruction': 0.2
        },
        metrics={
            'classification': ['accuracy', 'precision', 'recall'],
            'reconstruction': ['mse']
        }
    )
    
    return model