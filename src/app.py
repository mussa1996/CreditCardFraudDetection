# # app.py (FastAPI Main App)
# from fastapi import FastAPI
# from pydantic import BaseModel
# from predict import make_prediction

# app = FastAPI()

# class Transaction(BaseModel):
#     Time: float
#     V1: float
#     V2: float
#     V3: float
#     V4: float
#     V5: float
#     V6: float
#     V7: float
#     V8: float
#     V9: float
#     V10: float
#     V11: float
#     V12: float
#     V13: float
#     V14: float
#     V15: float
#     V16: float
#     V17: float
#     V18: float
#     V19: float
#     V20: float
#     V21: float
#     V22: float
#     V23: float
#     V24: float
#     V25: float
#     V26: float
#     V27: float
#     V28: float
#     Amount: float

# @app.post("/predict")
# def predict(transaction: Transaction):
#     transaction_dict = transaction.dict()
#     result = make_prediction(transaction_dict)
#     return result





#=============================================

# app.py (FastAPI Main App)
from fastapi import FastAPI
from pydantic import BaseModel
from predict import make_prediction, get_model_accuracy

app = FastAPI(title="Hybrid LSTM Fraud Detection API")

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def read_root():
    return {"message": "Hybrid LSTM Fraud Detection API", "status": "active"}

@app.post("/predict")
def predict(transaction: Transaction):
    transaction_dict = transaction.dict()
    result = make_prediction(transaction_dict)
    return result

@app.get("/accuracy")
def get_accuracy():
    """Get current model accuracy and statistics"""
    return get_model_accuracy()

@app.get("/buffer-status")
def get_buffer_status():
    """Get current sequence buffer status"""
    from model_loader import prediction_buffer
    return {
        "buffer_size": len(prediction_buffer.buffer),
        "required_size": prediction_buffer.sequence_length,
        "is_ready": prediction_buffer.is_ready()
    }