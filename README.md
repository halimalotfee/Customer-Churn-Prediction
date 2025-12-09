# Customer-Churn-Prediction – Telecom

Machine Learning · XGBoost · FastAPI · Streamlit · AWS Deployment

🚀 Project Overview

This project focuses on predicting customer churn for a telecom company using machine learning techniques.
It covers the full ML pipeline, including:

✔ Data exploration
✔ Preprocessing & feature engineering
✔ Model training and hyperparameter tuning
✔ Performance evaluation
✔ Real-time prediction API (FastAPI)
✔ Web interface (Streamlit)
✔ Deployment on AWS (EC2 + S3)

The goal is to provide a production-ready churn prediction system that business teams can use to identify high-risk customers and take retention actions.
├── data/                     # Dataset (Telco Customer Churn)
├── notebooks/
│   ├── 01_EDA.ipynb         # Exploratory analysis
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
├── src/
│   ├── preprocessing.py      # Feature engineering, encoding, scaling
│   ├── train.py              # Model training script
│   ├── inference.py          # Predict function
│   ├── model.pkl             # Saved model
├── api/
│   ├── main.py               # FastAPI server
│   ├── requirements.txt
├── app/
│   ├── app.py                # Streamlit interface
├── deployment/
│   ├── dockerfile
│   ├── deploy.sh
│   ├── instructions.md       # AWS EC2 setup
├── README.md

📊 Dataset

Source: Telco Customer Churn dataset (Kaggle)
Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset includes:

Customer demographics

Services subscribed (phone, internet, streaming, protection, etc.)

Contract types

Billing and monthly charges

Churn information (Yes/No)
