##Step 1: Dataset Loading & EDA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data_tsla.csv")

# Load dataset with correct date column
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'], index_col='timestamp')

# Quick check
print(df.head())
print(df.info())

# Plot closing price
plt.figure(figsize=(12,6))
plt.plot(df['close'], label='TSLA Close Price')
plt.title('TSLA Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

##Step 2: Feature Engineering & Preprocessing
# Moving averages
df['MA10'] = df['close'].rolling(10).mean()
df['MA50'] = df['close'].rolling(50).mean()

# Daily returns
df['Daily_Return'] = df['close'].pct_change()

# Drop NaN from rolling
df = df.dropna()

# Features and target
X = df[['open', 'high', 'low', 'volume', 'MA10', 'MA50', 'Daily_Return']]
y = df['close']

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

##Step 3: Train 3 Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Initialize models
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb = XGBRegressor(n_estimators=100, random_state=42)

# Train
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

##Step 4: Evaluate Models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate(model):
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    return rmse, mae, r2

for model in [lr, rf, xgb]:
    rmse, mae, r2 = evaluate(model)
    print(f"{model.__class__.__name__} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

##Step 5: Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# --------- RandomForest Hyperparameter Tuning ---------
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}

grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='neg_mean_squared_error')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print("Best RandomForest Params:", grid_rf.best_params_)

# Evaluate tuned RF
rmse, mae, r2 = evaluate(best_rf)
print(f"Tuned RandomForest -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")


# --------- XGBoost Hyperparameter Tuning ---------
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
}

grid_xgb = GridSearchCV(XGBRegressor(random_state=42), xgb_params, cv=3, scoring='neg_mean_squared_error')
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
print("Best XGBoost Params:", grid_xgb.best_params_)

# Evaluate tuned XGB
rmse, mae, r2 = evaluate(best_xgb)
print(f"Tuned XGBoost -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

##Step 6: Unsupervised Learning (Clustering)
from sklearn.cluster import KMeans

# Use features: daily return and 10-day MA for clustering
cluster_features = df[['Daily_Return', 'MA10']].dropna()

# Fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(cluster_features)

# Visualize clusters
import matplotlib.pyplot as plt
plt.scatter(df['Daily_Return'], df['MA10'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Daily Return')
plt.ylabel('MA10')
plt.title('KMeans Clustering of TSLA Stock')
plt.show()

##Step 7: Final Results & Findings
import joblib
joblib.dump(best_xgb, os.path.join(BASE_DIR, "saved_models", "tsla_model.pkl"))
print("✅ Best model saved to saved_models/tsla_model.pkl")