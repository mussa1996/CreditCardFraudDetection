
# Credit Card Fraud Detection Using Hybrid Deep Learning Approaches

This project addresses data imbalance in real-time credit card fraud detection by employing a hybrid deep learning model that combines a classification branch with an autoencoder branch for reconstruction.

## Project Structure

CreditCardFraudDetection/ ├── data/ │ ├── raw/ # Raw dataset (e.g., creditcard.csv) │ └── processed/ # Processed data files ├── models/ # Saved model weights ├── notebooks/ # Jupyter notebooks for EDA and experiments ├── src/ # Source code │ ├── init.py │ ├── config.py # Configuration parameters │ ├── data_preprocessing.py # Data loading and preprocessing functions │ ├── model.py # Hybrid deep learning model definition │ ├── train.py # Training script │ ├── evaluate.py # Evaluation script │ └── utils.py # Utility functions (e.g., plotting) ├── requirements.txt # Project dependencies └── README.md # Project overview and instructions

## How to Run

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
2. **Prepare your dataset:**

 Place your credit card fraud dataset (creditcard.csv) in the data/raw/ folder.

3.**Train the model:**

 python src/train.py or python -m src.train
4. **Evaluate the model:**
python src/evaluate.py or python -m src.evaluate
5. **Explore with Jupyter Notebooks:**
Use notebooks in the notebooks/ directory for additional exploratory data analysis (EDA) and experiments.
