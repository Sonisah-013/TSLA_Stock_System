# 📈 TSLA Stock Price Prediction System

> A complete Machine Learning project to predict Tesla (TSLA) stock closing prices using historical market data and advanced regression models.

---

## 🚀 Overview

This project implements an end-to-end Machine Learning pipeline for stock price prediction.  
It covers data analysis, feature engineering, model training, evaluation, hyperparameter tuning, and unsupervised learning.

The goal is to build a robust system capable of predicting TSLA stock closing prices based on historical features.

---

## 🎯 Key Highlights

- ✔ End-to-end ML pipeline implementation  
- ✔ Feature engineering using financial indicators  
- ✔ Multiple regression models comparison  
- ✔ Hyperparameter tuning using GridSearchCV  
- ✔ Clustering for stock behavior analysis  
- ✔ Production-ready project structure  

---


---

## 🛠️ Tech Stack

- **Programming:** Python  
- **Data Analysis:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn, XGBoost  
- **Model Persistence:** Joblib  
- **Deployment (Optional):** FastAPI  

---

## 🔍 Machine Learning Workflow

### 📊 1. Data Loading & Exploratory Data Analysis
- Loaded TSLA dataset (`data_tsla.csv`)
- Performed statistical analysis
- Visualized trends and correlations

### 🔧 2. Feature Engineering
- Moving Averages (MA10, MA50)
- Daily Returns
- Data cleaning and preprocessing

### 🤖 3. Model Training
Trained and compared:
- Linear Regression  
- Random Forest Regressor  
- XGBoost Regressor  

### 📈 4. Model Evaluation
Used key regression metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

### ⚙️ 5. Hyperparameter Tuning
- Applied GridSearchCV
- Optimized Random Forest and XGBoost models

### 🧩 6. Unsupervised Learning
- KMeans clustering
- Identified market volatility patterns

### 🏆 7. Final Results
- XGBoost performed best
- Feature engineering significantly improved accuracy
- Model saved for deployment

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/TSLA_Stock_System.git
cd TSLA_Stock_System