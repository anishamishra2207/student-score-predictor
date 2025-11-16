# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import os

# --- Configuration ---
DATA_FILE = "housing.csv"        # your dataset
MODEL_FILE = "house_model.pkl"
SCALER_FILE = "scaler.pkl"
HISTORY_FILE = "history.csv"

# --- Load Data ---
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found. Create it first with columns Area,Bedrooms,Age,Price")

df = pd.read_csv(DATA_FILE)
required_cols = {"Area", "Bedrooms", "Age", "Price"}
if not required_cols.issubset(set(df.columns)):
    raise ValueError(f"Dataset must contain columns: {required_cols}")

X = df[["Area", "Bedrooms", "Age"]].values
y = df["Price"].values

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# --- Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train Model ---
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --- Evaluate ---
y_pred_test = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
print(f"R2 (test): {r2:.4f}")
print(f"MSE (test): {mse:.4f}")

# --- Save artifacts ---
pickle.dump(model, open(MODEL_FILE, "wb"))
pickle.dump(scaler, open(SCALER_FILE, "wb"))

# --- Create history file if not exists ---
if not os.path.exists(HISTORY_FILE):
    history = pd.DataFrame(columns=["Area", "Bedrooms", "Age", "PredictedPrice"])
    history.to_csv(HISTORY_FILE, index=False)

print(f"\nSaved model -> {MODEL_FILE}")
print(f"Saved scaler -> {SCALER_FILE}")
print(f"History file ready -> {HISTORY_FILE}")
