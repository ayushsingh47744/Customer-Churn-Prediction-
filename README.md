# 📊 Customer Retention Analytics Dashboard

An end-to-end machine learning project that predicts customer churn for a bank and presents the results through an interactive business analytics dashboard.

🔗 **Live App:** [customer-retention-analytics-2026.streamlit.app](https://customer-retention-analytics-2026.streamlit.app)

---

## 🧠 Overview

This project analyzes real-world bank customer data to identify patterns behind customer churn and predict which customers are likely to leave. It combines a complete ML pipeline (data preprocessing, feature engineering, model comparison, and evaluation) with a business-facing Streamlit dashboard for exploring insights and running live predictions.

---

## ✨ Features

- **📈 Overview** — Key churn KPIs, revenue-at-risk estimate, churn trends by country and age
- **👥 Customer Segments** — Breakdown by age, product count, salary, and gender
- **⚠️ Risk Analysis** — What drives churn: activity status, credit card ownership, balance, tenure, and feature correlations
- **💡 Retention Insights** — Auto-generated business recommendations based on the data
- **🔮 Predict Churn** — Live prediction tool for new or existing customers with a visual probability gauge

---

## 🧩 Tech Stack

- **Python**
- **Pandas / NumPy** — data processing
- **Scikit-learn / XGBoost** — model training and evaluation
- **Plotly** — interactive visualizations
- **Streamlit** — dashboard and deployment

---

## 📂 Project Structure

```
├── app.py                     # Streamlit dashboard (main app)
├── retrain.py                 # Standalone script to retrain and save the model
├── Analysis.ipynb             # Full exploratory analysis + model training notebook
├── customer_data.csv          # Dataset
├── best_churn_pipeline.pkl    # Trained model pipeline
├── requirements.txt           # Python dependencies
└── README.md
```

---

## ⚙️ Running Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/tripathipalak/Customer-churn-prediction.git
   cd Customer-churn-prediction
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

5. Open your browser at `http://localhost:8501`

---

## 🔄 Retraining the Model

If you update the dataset or want to retrain the model with your local environment's library versions, run:

```bash
python retrain.py
```

This regenerates `best_churn_pipeline.pkl` using the current `customer_data.csv`.

---

## 📊 Model Performance

The final model (GradientBoosting) was selected based on cross-validated ROC AUC across 5 candidate models:

| Metric      | Score  |
|-------------|--------|
| Accuracy    | ~0.86  |
| Precision   | ~0.77  |
| Recall      | ~0.48  |
| F1-score    | ~0.59  |
| ROC AUC     | ~0.87  |

---

## 📌 Dataset

The dataset contains 10,000 bank customer records with features including credit score, geography, gender, age, tenure, balance, number of products, credit card ownership, active membership status, and estimated salary.

---

## 📝 License

This project is for educational and portfolio purposes.
