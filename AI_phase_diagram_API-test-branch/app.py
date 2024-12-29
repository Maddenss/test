# app.py
import streamlit as st
import requests
import pandas as pd

# URL FastAPI сервера (если FastAPI и Streamlit находятся на одной машине, используйте localhost)
API_URL = "http://localhost:8000"

# Заголовок приложения
st.title("Prediction and Model Analysis")

# Боковая панель для загрузки файла данных
st.sidebar.header("Upload Data for Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV file with columns: T, E, C, FM, Xfm, AFM, Xafm", type="csv")

# Покажем данные, если файл загружен
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", data)

    # Отправка данных для предсказания
    if st.button("Predict"):
        files = {"file": uploaded_file}
        response = requests.post(f"{API_URL}/predict", files=files)

        if response.status_code == 200:
            predictions = pd.DataFrame(response.json())
            st.write("Predictions:", predictions)
        else:
            st.error(f"Error: {response.text}")

# Метрики модели
if st.button("Show Metrics"):
    response = requests.get(f"{API_URL}/metrics")
    if response.status_code == 200:
        metrics = response.json()
        st.write("Model Metrics:", metrics)
    else:
        st.error(f"Error: {response.text}")

# ROC Curve
if st.button("Show ROC Curve"):
    response = requests.get(f"{API_URL}/roc-curve")
    if response.status_code == 200:
        st.image(response.content)
    else:
        st.error(f"Error: {response.text}")

# Кросс-валидация: результаты
if st.button("Show Cross-Validation Results"):
    response = requests.get(f"{API_URL}/cross-validation")
    if response.status_code == 200:
        results = response.json()
        st.write("Cross-Validation Results:", results)
    else:
        st.error(f"Error: {response.text}")

# Кросс-валидация с визуализацией ROC-AUC
if st.button("Show Cross-Validation ROC-AUC Plot"):
    response = requests.get(f"{API_URL}/cross-validation/plot")
    if response.status_code == 200:
        st.image(response.content)
    else:
        st.error(f"Error: {response.text}")

# Важность признаков
if st.button("Show Feature Importance"):
    response = requests.get(f"{API_URL}/feature-importance")
    if response.status_code == 200:
        st.image(response.content)
    else:
        st.error(f"Error: {response.text}")

# SHAP-анализ
if st.button("Show SHAP Analysis"):
    response = requests.get(f"{API_URL}/shap-analysis")
    if response.status_code == 200:
        st.image(response.content)
    else:
        st.error(f"Error: {response.text}")
