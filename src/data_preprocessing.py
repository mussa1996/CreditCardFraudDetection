# src/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.config import RAW_DATA_PATH, TEST_SIZE, RANDOM_STATE

def load_data():
    """
    Load the credit card fraud dataset.
    """
    df = pd.read_csv(RAW_DATA_PATH)
    return df

def preprocess_data(df):
    """
    Preprocess the dataset by separating features and the target.
    Assumes 'Class' is the target column.
    """
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y

def handle_imbalance(X, y):
    """
    Use SMOTE to oversample the minority class.
    """
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def split_data(X, y):
    """
    Split the data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data()
    X, y = preprocess_data(df)
    print("Before balancing:", y.value_counts())
    X_res, y_res = handle_imbalance(X, y)
    print("After balancing:", pd.Series(y_res).value_counts())
