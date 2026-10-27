📊 Customer Churn Prediction & Analytics Dashboard

An end-to-end Machine Learning project that analyzes customer behavior, predicts customer churn, and provides actionable insights to help businesses improve customer retention.

The project combines a trained machine learning pipeline with an interactive Streamlit dashboard to explore customer segments, identify churn patterns, analyze potential risk factors, and predict whether a customer is likely to leave.

🚀 Project Overview

Customer churn occurs when customers stop using a company's products or services. Understanding churn patterns helps businesses identify at-risk customers and develop better retention strategies.

This project uses historical customer data and machine learning to predict churn while providing interactive visualizations for business analysis.

✨ Features

- Customer Churn Prediction: Predict customer churn using a trained machine learning pipeline.
- Interactive Dashboard: Explore customer analytics through a Streamlit interface.
- Business Overview: View total customers, churned customers, churn rate, average balance, and estimated revenue at risk.
- Customer Segmentation: Analyze churn across age groups, gender, country, and number of products.
- Risk Analysis: Examine churn patterns based on membership activity and credit card ownership.
- Retention Insights: Explore customer behavior to support retention strategies.
- Interactive Filters: Filter customer data by country, gender, and membership status.
- Data Visualization: Generate interactive charts using Plotly.
- Model Retraining: Use the provided retraining script to update the machine learning pipeline.

🛠️ Tech Stack

Category| Technologies
Programming Language| Python
Data Analysis| Pandas, NumPy
Machine Learning| Scikit-learn
Model Persistence| Joblib
Dashboard| Streamlit
Data Visualization| Plotly
Development Environment| Jupyter Notebook
Dataset Format| CSV

📁 Project Structure

Customer-Churn-Prediction-/
│
├── Analysis.ipynb             # Exploratory data analysis
├── app.py                     # Streamlit dashboard
├── customer_data.csv          # Customer dataset
├── best_churn_pipeline.pkl    # Trained ML pipeline
├── retrain.py                 # Model retraining script
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation

📈 Dashboard Sections

1. Overview

- Total number of customers
- Churned customers and overall churn rate
- Estimated revenue at risk
- Average customer balance
- Churn trends across countries and age groups

2. Customer Segmentation

- Customer age distribution by churn status
- Churn rate by number of products
- Salary distribution across churn categories
- Gender-based churn analysis
- Customer segment summary tables

3. Risk Analysis

- Churn rate among active and inactive members
- Comparison of customers with and without credit cards
- Analysis of customer characteristics associated with churn

4. Retention Insights

Provides business-oriented insights from customer data to help understand churn behavior and identify potential retention opportunities.

5. Predict Churn

Uses the saved machine learning pipeline to generate churn predictions from customer information entered through the dashboard.

⚙️ Installation and Setup

Follow these steps to run the project locally.

Prerequisites

- Python 3.9 or later
- pip package manager
- Git

Step 1: Clone the Repository

git clone https://github.com/ayushsingh47744/Customer-Churn-Prediction-.git

Step 2: Navigate to the Project Directory

cd Customer-Churn-Prediction-

Step 3: Create a Virtual Environment (Recommended)

python -m venv venv

Activate the environment.

Windows:

venv\Scripts\activate

macOS/Linux:

source venv/bin/activate

Step 4: Install Dependencies

pip install -r requirements.txt

Step 5: Run the Dashboard

streamlit run app.py

The application will open in your default browser. If it does not, open the local URL displayed in your terminal.

Important: Keep "customer_data.csv" and "best_churn_pipeline.pkl" in the project directory, as the application loads these files when it starts.

🧠 Machine Learning Workflow

The project follows a typical machine learning workflow:

1. Data Exploration: Examine customer data and identify relevant patterns.
2. Data Preprocessing: Prepare customer features for model training.
3. Model Training: Train a machine learning model to predict churn.
4. Model Evaluation: Assess the model's predictive performance.
5. Model Persistence: Save the trained pipeline for reuse.
6. Prediction: Load the saved pipeline and generate predictions for customer inputs.
7. Visualization: Present customer analytics and churn patterns through an interactive dashboard.

🎯 Project Objectives

- Predict customer churn using machine learning.
- Understand the factors associated with customer attrition.
- Analyze churn patterns across different customer segments.
- Present complex customer data through interactive visualizations.
- Help businesses make data-informed customer retention decisions.

🔮 Future Improvements

- Compare multiple machine learning algorithms.
- Add model evaluation metrics such as precision, recall, F1-score, and ROC-AUC.
- Improve prediction explanations using feature importance or SHAP.
- Deploy the dashboard to a cloud platform.
- Introduce automated model monitoring and periodic retraining.
- Develop personalized retention recommendations for high-risk customers.

💡 Learning Outcomes

Through this project, I explored:

- Data analysis and preprocessing using Python.
- Machine learning pipeline usage and model persistence.
- Interactive dashboard development with Streamlit.
- Customer segmentation and churn analysis.
- Data visualization with Plotly.
- Integrating a trained ML model into a user-facing application.

👨‍💻 Author

Ayush Singh

GitHub: "@ayushsingh47744" (https://github.com/ayushsingh47744)

---

⭐ If you find this project useful, consider giving the repository a star.
