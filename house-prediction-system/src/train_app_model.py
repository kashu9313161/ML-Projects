import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

from src.preprocessing import build_preprocessor
from src.predict import predict_price


DATA_PATH = "data/raw/AmesHousing.csv"
MODEL_PATH = "models/house_price_app_xgb.joblib"

RANDOM_STATE = 42


APP_FEATURES = [
    "Overall Qual",
    "Overall Cond",
    "Gr Liv Area",
    "1st Flr SF",
    "2nd Flr SF",
    "Total Bsmt SF",
    "Garage Cars",
    "Garage Area",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "TotRms AbvGrd",
    "Year Built",
    "Year Remod/Add",
    "Neighborhood",
    "Kitchen Qual",
    "Exter Qual",
    "Lot Area",
    "Fireplaces"
]


# --------------------------------
# 1. Load data
# --------------------------------

df = pd.read_csv(DATA_PATH)

print("Original dataset:", df.shape)


# --------------------------------
# 2. Select application features
# --------------------------------

X = df[APP_FEATURES].copy()

y = np.log1p(df["SalePrice"])


print("Application features:", X.shape)
print("Number of features:", len(APP_FEATURES))


# --------------------------------
# 3. Train-test split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)


# --------------------------------
# 4. Preprocessing
# --------------------------------

preprocessor, numerical_features, categorical_features = build_preprocessor(
    X_train
)


# --------------------------------
# 5. XGBoost
# --------------------------------

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


# --------------------------------
# 6. Complete pipeline
# --------------------------------

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", model)
])


# --------------------------------
# 7. Train
# --------------------------------

print("\nTraining application model...")

pipeline.fit(X_train, y_train)

print("Training complete.")


# --------------------------------
# 8. Prediction
# --------------------------------

y_pred_log = pipeline.predict(X_test)


# Convert log price → original price

y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred_log)


# --------------------------------
# 9. Evaluation
# --------------------------------

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


print("\n========== APPLICATION MODEL ==========")

print(f"MAE  : ${mae:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# --------------------------------
# 10. Save model
# --------------------------------

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

joblib.dump(
    pipeline,
    MODEL_PATH
)

print(f"\nModel saved to: {MODEL_PATH}")

# --------------------------------
# 11. Feature Importance
# --------------------------------

regressor = pipeline.named_steps["regressor"]

feature_names = (
    pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importance = regressor.feature_importances_

feature_importance = (
    pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })
    .sort_values("Importance", ascending=False)
)

print("\n========== TOP FEATURES ==========")

print(
    feature_importance.head(20).to_string(index=False)
)