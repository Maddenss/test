# AI Phase Diagram API

## Описание:
Это приложение реализует API для выполнения различных задач машинного обучения, включая:
- Расчет метрик модели.
- Генерацию ROC-кривых.
- Кросс-валидацию.
- SHAP-анализ для интерпретации моделей.
- Получение важности признаков.
- Выполнение предсказаний на основе загружаемых данных.


## Установка и запуск:

1. Установите зависимости: `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

2. Запустите сервер FastAPI:
    ```bash
    uvicorn app:app --reload
    ```

3. Перейдите в браузере по адресу:
    ```bash
    http://127.0.0.1:8000/docs
    ```


## API Функции

### **GET /metrics**
Возвращает метрики модели, обученной на тестовых данных, включая:

- **Accuracy**: Точность классификации.
- **Precision**: Точность для классов 0 и 1.
- **Recall**: Полнота для классов 0 и 1.
- **F1-score**: F1-метрика для каждого класса.
- **ROC-AUC**: Площадь под кривой ошибок классификации.

---

### **GET /roc-curve**
Генерирует и возвращает изображение **ROC-кривой**, визуализируя производительность модели.

---

### **GET /cross-validation**
Проводит 5-фолд кросс-валидацию на данных и возвращает результаты:

- Матрица ошибок (**Confusion Matrix**) для каждого фолда.
- Значение **ROC-AUC** для каждого фолда.

---

### **GET /cross-validation/plot**
Генерирует и возвращает график значений **ROC-AUC** для всех фолдов.

---

### **GET /feature-importance**
Возвращает график важности признаков модели, выполненный с использованием **SHAP**.

---

### **GET /shap-analysis**
Генерирует подробную интерпретацию модели с использованием **SHAP** и визуализирует вклад каждого признака в предсказания.

---

### **POST /predict**
Позволяет загрузить **CSV-файл** с данными для выполнения предсказаний:

**Требования к файлу**:
- Должны присутствовать следующие столбцы:
  - `T`, `E`, `C`, `FM`, `Xfm`, `AFM`, `Xafm`.
- **Формат файла**: CSV.
