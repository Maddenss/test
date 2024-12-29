import streamlit as st
import matplotlib.pyplot as plt
import shap
import time
from main import (
    load_data,
    load_model,
    train_test_split_data,
    calculate_metrics,
)

MODEL_PATH = "xgboost_model.pkl"
DATA_PATH = "Main_Data.csv"

@st.cache_data
def cached_load_data(data_path):
    return load_data(data_path)

@st.cache_data
def cached_load_model(model_path):
    return load_model(model_path)

st.title("AI_phase_diagram_API")

X, y = cached_load_data(DATA_PATH)
model = cached_load_model(MODEL_PATH)
X_train, X_test, y_train, y_test = train_test_split_data(X, y)

st.header("Метрики модели")
if st.button("Рассчитать метрики"):
    with st.spinner("Расчет метрик модели..."):
        start_time = time.time()
        metrics = calculate_metrics(model, X_test, y_test)
        st.json(metrics)
        elapsed_time = time.time() - start_time
        st.write(f"Время выполнения: {elapsed_time:.2f} секунд")
