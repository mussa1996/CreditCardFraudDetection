# src/evaluate_comprehensive.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import tensorflow as tf
from src.data_preprocessing import load_data
from src.sequence_preprocessing import prepare_sequential_data
from src.config import MODEL_SAVE_PATH, SEQUENCE_LENGTH, FRAUD_THRESHOLD

def evaluate_comprehensive():
    """Comprehensive evaluation of the hybrid model"""
    
    print("Loading data and model...")
    # Load data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_sequential_data(
        df, sequence_length=SEQUENCE_LENGTH
    )
    
    # Load model
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    
    print("Making predictions...")
    # Get predictions
    predictions, reconstructions = model.predict(X_test, verbose=1)
    y_pred = (predictions > FRAUD_THRESHOLD).astype(int).flatten()
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    auc_score = roc_auc_score(y_test, predictions)
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Test Set Size: {len(y_test)} sequences")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"{'='*60}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, predictions)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Prediction distribution
    plt.figure(figsize=(10, 6))
    plt.hist(predictions[y_test == 0], bins=50, alpha=0.7, label='Non-Fraud', color='green')
    plt.hist(predictions[y_test == 1], bins=50, alpha=0.7, label='Fraud', color='red')
    plt.axvline(x=FRAUD_THRESHOLD, color='black', linestyle='--', label=f'Threshold ({FRAUD_THRESHOLD})')
    plt.title('Distribution of Prediction Scores')
    plt.xlabel('Fraud Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'auc_score': auc_score,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

if __name__ == "__main__":
    results = evaluate_comprehensive()
    
    # Save results to file
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'AUC Score'],
        'Value': [results['accuracy'], results['auc_score']]
    })
    results_df.to_csv('results/evaluation_metrics.csv', index=False)
    print("\nEvaluation results saved to 'results/evaluation_metrics.csv'")