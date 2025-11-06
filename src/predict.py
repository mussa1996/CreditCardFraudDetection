# # predict.py
# import pandas as pd
# import os
# from model_loader import model, FEATURE_COLUMNS

# LOG_PATH = "results/logged_predictions.csv"
# THRESHOLD = 0.5

# def make_prediction(transaction_dict):
#     try:
#         df = pd.DataFrame([transaction_dict])
#         df = df[FEATURE_COLUMNS]  # Ensure correct order

#         prediction, _ = model.predict(df, verbose=0)
#         score = prediction[0][0]
#         is_fraud = bool(score > THRESHOLD)

#         df["score"] = score
#         df["is_fraud"] = is_fraud
#         write_header = not os.path.exists(LOG_PATH)
#         df.to_csv(LOG_PATH, mode="a", header=write_header, index=False)

#         return {"score": round(float(score), 4), "is_fraud": is_fraud}
    
#     except Exception as e:
#         print("❌ Prediction error:", e)
#         return {"error": str(e)}



#=============================================
# predict.py
import pandas as pd
import numpy as np
import os
from model_loader import model, prediction_buffer, FEATURE_COLUMNS, FRAUD_THRESHOLD

LOG_PATH = "results/logged_predictions.csv"

def make_prediction(transaction_dict):
    try:
        # Add transaction to sequence buffer
        prediction_buffer.add_transaction(transaction_dict)
        
        # Check if we have enough data for prediction
        if not prediction_buffer.is_ready():
            return {
                "status": "insufficient_data",
                "message": f"Need {prediction_buffer.sequence_length} transactions, have {len(prediction_buffer.buffer)}",
                "score": 0.0,
                "is_fraud": False
            }
        
        # Get sequence and make prediction
        sequence = prediction_buffer.get_sequence()
        prediction, reconstruction = model.predict(sequence, verbose=0)
        
        score = float(prediction[0][0])
        is_fraud = bool(score > FRAUD_THRESHOLD)
        
        # Log the prediction
        log_prediction(transaction_dict, score, is_fraud)
        
        return {
            "status": "success",
            "score": round(score, 4),
            "is_fraud": is_fraud,
            "confidence": "high" if abs(score - 0.5) > 0.3 else "medium",
            "sequence_ready": True,
            "transactions_in_buffer": len(prediction_buffer.buffer)
        }
    
    except Exception as e:
        print("❌ Prediction error:", e)
        return {"status": "error", "error": str(e)}

def log_prediction(transaction_dict, score, is_fraud):
    """Log prediction results to CSV"""
    log_data = transaction_dict.copy()
    log_data["score"] = score
    log_data["is_fraud"] = is_fraud
    log_data["timestamp"] = pd.Timestamp.now()
    
    df_log = pd.DataFrame([log_data])
    
    write_header = not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0
    df_log.to_csv(LOG_PATH, mode="a", header=write_header, index=False)

def get_model_accuracy():
    """Calculate and return current model accuracy from logs"""
    try:
        if not os.path.exists(LOG_PATH):
            return {"accuracy": 0.0, "message": "No predictions logged yet"}
        
        df = pd.read_csv(LOG_PATH)
        if len(df) < 10:  # Need minimum samples
            return {"accuracy": 0.0, "message": "Insufficient prediction data"}
        
        # Calculate accuracy metrics
        total_predictions = len(df)
        fraud_predictions = df['is_fraud'].sum()
        non_fraud_predictions = total_predictions - fraud_predictions
        
        # For demonstration, if we had ground truth we'd calculate real accuracy
        # Since we don't have ground truth in production, we return basic stats
        accuracy_stats = {
            "total_predictions": total_predictions,
            "fraud_predictions": int(fraud_predictions),
            "non_fraud_predictions": int(non_fraud_predictions),
            "fraud_rate": round((fraud_predictions / total_predictions) * 100, 2),
            "average_confidence": round(df['score'].mean(), 4)
        }
        
        return accuracy_stats
        
    except Exception as e:
        return {"error": f"Accuracy calculation failed: {str(e)}"}