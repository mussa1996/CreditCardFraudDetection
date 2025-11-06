# simulate_transactions.py
import pandas as pd
import requests
import time
import os

# Load dataset
# df = pd.read_csv("data/raw/creditcard.csv")
# X = df.drop("Class", axis=1)
# Construct absolute path to the dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)
X = df.drop("Class", axis=1)

# Simulate sending transactions one by one
for i in range(100):  # Simulate first 100 transactions
    transaction = X.iloc[i].to_dict()
    response = requests.post("http://127.0.0.1:9000/predict", json=transaction)
    if response.status_code == 200:
        print(f"Transaction {i+1}:", response.json())
    else:
        print(f"Transaction {i+1}: Failed with status {response.status_code}, body:", response.text)
    time.sleep(0.5)  # Simulate time delay (500ms) between transactions
