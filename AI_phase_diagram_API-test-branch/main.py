import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

MODEL_PATH = "xgboost_model.pkl"
DATA_PATH = "Main_Data.csv"


def load_data(data_path):
    data = pd.read_csv(data_path)
    X = data[["T", "E", "C", "FM", "Xfm", "AFM", "Xafm"]]
    y = data["h"]
    return X, y


def load_model(model_path):
    with open(model_path, "rb") as file:
        return pickle.load(file)


def train_test_split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)


def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (Class 0)": precision_score(y_test, y_pred, pos_label=0),
        "Precision (Class 1)": precision_score(y_test, y_pred, pos_label=1),
        "Recall (Class 0)": recall_score(y_test, y_pred, pos_label=0),
        "Recall (Class 1)": recall_score(y_test, y_pred, pos_label=1),
        "F1-score (Class 0)": f1_score(y_test, y_pred, pos_label=0),
        "F1-score (Class 1)": f1_score(y_test, y_pred, pos_label=1),
        "ROC-AUC": roc_auc_score(y_test, y_pred_prob),
    }
    return metrics
