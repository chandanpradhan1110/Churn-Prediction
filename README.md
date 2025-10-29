📊 Customer Churn Prediction
🧩 Project Overview

Customer churn refers to the loss of clients or subscribers — in other words, when customers stop doing business with a company. Predicting churn helps businesses identify at-risk customers and take proactive steps to retain them.

This project aims to build a predictive machine learning model to classify whether a customer is likely to churn based on historical behavior and account information.

🎯 Objectives

Identify key factors influencing customer churn.

Develop and evaluate machine learning models to predict churn.

Provide actionable insights to reduce churn rate and improve customer retention.

🗂️ Dataset

The dataset used in this project contains customer demographic, account, and usage details.

Example Columns:
Feature	Description
customerID	Unique ID for each customer
gender	Male or Female
SeniorCitizen	Whether the customer is a senior citizen (1/0)
Partner	Whether the customer has a partner (Yes/No)
Dependents	Whether the customer has dependents
tenure	Number of months the customer has stayed with the company
PhoneService	Whether the customer has phone service
InternetService	Type of internet service (DSL, Fiber optic, None)
MonthlyCharges	The amount charged to the customer monthly
TotalCharges	Total amount charged
Churn	Target variable (Yes/No)
⚙️ Technologies Used

Programming Language: Python 🐍

Libraries:

pandas, numpy — Data preprocessing

matplotlib, seaborn — Data visualization

scikit-learn — Machine learning modeling

xgboost, lightgbm — Advanced models

joblib — Model saving/loading

🧠 Methodology
1. Exploratory Data Analysis (EDA)

Checked for null values and handled missing data.

Visualized churn distribution.

Analyzed relationships between features and churn rate.

2. Data Preprocessing

Encoded categorical variables using One-Hot Encoding.

Scaled numerical features using StandardScaler.

Split dataset into training and testing sets (e.g., 80-20).

3. Model Development

Trained multiple models including:

Logistic Regression

Random Forest

XGBoost

LightGBM

Evaluated models using metrics:

Accuracy

Precision

Recall

F1-Score

ROC-AUC

4. Model Evaluation

Used confusion matrix and ROC curve for visual evaluation.
Best performing model was selected based on F1-Score and AUC.

5. Model Deployment (Optional)

Model can be deployed as a Flask API or Streamlit Web App for real-time churn prediction.

📈 Results

Achieved an accuracy of ~85-90% (depending on dataset).

Identified key churn indicators such as:

Tenure (short-term customers are more likely to churn).

Contract type (month-to-month customers have higher churn).

Monthly charges (high-charging customers are more likely to churn).

💡 Insights & Recommendations

Offer loyalty discounts for short-tenure customers.

Promote long-term contracts to reduce churn.

Improve customer support for high-charging customers.

🧰 How to Run

Clone this repository:

git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction


Install dependencies:

pip install -r requirements.txt


Run the Jupyter Notebook:

jupyter notebook churn_prediction.ipynb


(Optional) Run Web App:

streamlit run app.py

📂 Project Structure
churn-prediction/
│
├── data/
│   └── churn_data.csv
│
├── notebooks/
│   └── churn_prediction.ipynb
│
├── models/
│   └── churn_model.pkl
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
