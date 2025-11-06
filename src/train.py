# # src/train.py
# import pandas as pd
# from src.data_preprocessing import load_data, preprocess_data, handle_imbalance, split_data
# from src.model import create_hybrid_model
# from src.config import MODEL_SAVE_PATH, EPOCHS, BATCH_SIZE
# import numpy as np
# from src.progress_callback import TrainingProgressCallback

# def train():
#     # Load and preprocess data
#     df = load_data()
#     X, y = preprocess_data(df)
#     X_res, y_res = handle_imbalance(X, y)
    
#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = split_data(X_res, y_res)
    
#     # Create the hybrid model
#     input_dim = X_train.shape[1]
#     model = create_hybrid_model(input_dim)

#     # Set up the custom progress callback
#     progress_callback = TrainingProgressCallback()
    
#     # Train the model
#     history = model.fit(
#         X_train, 
#         {'classification': y_train, 'reconstruction': X_train},
#         validation_data=(X_test, {'classification': y_test, 'reconstruction': X_test}),
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         callbacks=[progress_callback]
#     )
    
#     # Save the trained model
#     model.save(MODEL_SAVE_PATH)
#     print("Model saved at:", MODEL_SAVE_PATH)
    
#     return history

# if __name__ == "__main__":
#     history =train()
#     print("Training completed.")
#     print("Model training and saving process finished successfully.")
#    # Display a summary for all epochs
#     print("\nSummary for all epochs:")
#     for epoch in range(len(history.history['loss'])):
#         epoch_num = epoch + 1
#         loss = history.history['loss'][epoch]
#         acc = history.history['classification_accuracy'][epoch] if 'classification_accuracy' in history.history else "N/A"
#         print(f"Epoch {epoch_num}: Loss = {loss:.4f}, Classification Accuracy = {acc}")
    



# ========================================


# src/train.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_preprocessing import load_data
from src.sequence_preprocessing import prepare_sequential_data
from src.model import create_hybrid_model, create_simple_hybrid_model
from src.config import MODEL_SAVE_PATH, EPOCHS, BATCH_SIZE, SEQUENCE_LENGTH
from src.progress_callback import TrainingProgressCallback

def plot_training_history(history):
    """Plot training history for accuracy and loss"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Classification accuracy
    axes[0, 0].plot(history.history['classification_accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_classification_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Classification Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Classification loss
    axes[0, 1].plot(history.history['classification_loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_classification_loss'], label='Validation Loss')
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Reconstruction loss
    axes[1, 0].plot(history.history['reconstruction_loss'], label='Training Loss')
    axes[1, 0].plot(history.history['val_reconstruction_loss'], label='Validation Loss')
    axes[1, 0].set_title('Reconstruction Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Overall loss
    axes[1, 1].plot(history.history['loss'], label='Training Loss')
    axes[1, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1, 1].set_title('Overall Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def train():
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Prepare sequential data
    print("Preparing sequential data...")
    X_train, X_test, y_train, y_test, scaler = prepare_sequential_data(
        df, 
        sequence_length=SEQUENCE_LENGTH, 
        test_size=0.2
    )
    
    print(f"Training sequences: {X_train.shape}")
    print(f"Testing sequences: {X_test.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Testing labels: {y_test.shape}")
    
    # Create model
    print("Creating hybrid LSTM + Autoencoder model...")
    input_dim = X_train.shape[2]  # Number of features
    model = create_hybrid_model(input_dim, SEQUENCE_LENGTH)
    # model = create_simple_hybrid_model(input_dim, SEQUENCE_LENGTH)  # For faster training
    
    # Set up callbacks
    progress_callback = TrainingProgressCallback()
    
    # For the reconstruction target, we use the last transaction of each sequence
    # This helps the autoencoder learn to reconstruct normal transaction patterns
    X_train_reconstruction = X_train[:, -1, :]  # Last transaction in each sequence
    X_test_reconstruction = X_test[:, -1, :]
    
    print("Starting training...")
    # Train the model
    history = model.fit(
        X_train, 
        {
            'classification': y_train, 
            'reconstruction': X_train_reconstruction
        },
        validation_data=(
            X_test, 
            {
                'classification': y_test, 
                'reconstruction': X_test_reconstruction
            }
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[progress_callback],
        verbose=0  # We use our custom callback for progress
    )
    
    # Save the model
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved at: {MODEL_SAVE_PATH}")
    
    # Plot training history
    plot_training_history(history)
    
    # Print final accuracy
    final_train_acc = history.history['classification_accuracy'][-1]
    final_val_acc = history.history['val_classification_accuracy'][-1]
    
    print(f"\n{'='*50}")
    print(f"TRAINING RESULTS SUMMARY:")
    print(f"{'='*50}")
    print(f"Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"{'='*50}")
    
    return history, model

if __name__ == "__main__":
    history, model = train()
    print("Training completed successfully!")