# 📊 Customer-Churn-Prediction – Telecom

Machine Learning · XGBoost · FastAPI · Streamlit · AWS Deployment

---

## 🚀 Project Overview

This project focuses on predicting customer churn for a telecom company using machine learning techniques.  
It covers the full ML pipeline, including:

- ✔️ Data exploration  
- ✔️ Preprocessing & feature engineering  
- ✔️ Model training and hyperparameter tuning  
- ✔️ Performance evaluation  
- ✔️ Real-time prediction API (FastAPI)  
- ✔️ Web interface (Streamlit)  
- ✔️ Deployment on AWS (EC2 + S3)

---

## 📁 Project Structure

├── data/ # Dataset (Telco Customer Churn)
├── notebooks/
│ ├── 01_EDA.ipynb # Exploratory analysis
│ ├── 02_Preprocessing.ipynb
│ ├── 03_Model_Training.ipynb
├── src/
│ ├── preprocessing.py # Feature engineering, encoding, scaling
│ ├── train.py # Model training script
│ ├── inference.py # Predict function
│ ├── model.pkl # Saved model
├── api/
│ ├── main.py # FastAPI server
│ ├── requirements.txt
├── app/
│ ├── app.py # Streamlit interface
├── deployment/
│ ├── dockerfile
│ ├── deploy.sh
│ ├── instructions.md # AWS deployment guide
