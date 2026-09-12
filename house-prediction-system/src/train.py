import os

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

from feature_engineering import create_features
from preprocessing import build_preprocessor


# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = "data/raw/AmesHousing.csv"
MODEL_PATH = "models/house_price_xgb.joblib"

RANDOM_STATE = 42


# -----------------------------
# Load Data
# -----------------------------

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# -----------------------------
# Feature Engineering
# -----------------------------

df = create_features(df)

print(f"After feature engineering: {df.shape}")


# -----------------------------
# Target
# -----------------------------

y = np.log1p(df["SalePrice"])

X = df.drop(columns=["SalePrice"])


# -----------------------------
# Train / Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print(f"Training data: {X_train.shape}")
print(f"Testing data:  {X_test.shape}")


# -----------------------------
# Preprocessor
# -----------------------------

preprocessor, numerical_features, categorical_features = (
    build_preprocessor(X_train)
)

print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")


# -----------------------------
# XGBoost
# -----------------------------

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE
)


# -----------------------------
# Complete Pipeline
# -----------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ]
)


# -----------------------------
# Train
# -----------------------------

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Training complete.")


# -----------------------------
# Prediction
# -----------------------------

y_pred_log = pipeline.predict(X_test)


# Convert back to dollars
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred_log)


# -----------------------------
# Evaluation
# -----------------------------

mae = mean_absolute_error(
    y_test_original,
    y_pred_original
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_original,
        y_pred_original
    )
)

r2 = r2_score(
    y_test_original,
    y_pred_original
)


print("\n========== MODEL PERFORMANCE ==========")

print(f"MAE  : ${mae:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# -----------------------------
# Save Model
# -----------------------------

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

joblib.dump(
    pipeline,
    MODEL_PATH
)

print(f"\nModel saved to: {MODEL_PATH}")