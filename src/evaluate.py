import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from src.data_preprocessing import load_data, preprocess_data, handle_imbalance, split_data
from src.config import MODEL_SAVE_PATH, BATCH_SIZE
from src.utils import plot_confusion_matrix

def evaluate():
    # Load and preprocess data
    df = load_data()
    X, y = preprocess_data(df)
    X_res, y_res = handle_imbalance(X, y)
    X_train, X_test, y_train, y_test = split_data(X_res, y_res)

    # Load the saved model
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    # model = tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False)

    # Get predictions from the classification branch
    predictions, _ = model.predict(X_test, batch_size=BATCH_SIZE)
    y_pred = (predictions > 0.5).astype(int)

    # Classification report as pandas DataFrame
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("\nClassification Report:\n", report_df)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # Plot confusion matrix
    plot_confusion_matrix(cm, classes=['Non-Fraud', 'Fraud'])

    # Optional: Plot histogram of prediction probabilities
    plt.figure(figsize=(8, 4))
    plt.hist(predictions, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of Prediction Probabilities')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    evaluate()
