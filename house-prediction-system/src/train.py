import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import build_preprocessor
from sklearn.ensemble import RandomForestRegressor


def train_model():

    # -----------------------------
    # Load dataset
    # -----------------------------

    df = pd.read_csv(
        "data/raw/AmesHousing.csv"
    )

    # -----------------------------
    # Separate features and target
    # -----------------------------

    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    # -----------------------------
    # Train / Test Split
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training features:", X_train.shape)
    print("Testing features :", X_test.shape)

    # -----------------------------
    # Build preprocessor
    # -----------------------------

    preprocessor, _, _ = build_preprocessor(X_train)

    # -----------------------------
    # Linear Regression Pipeline
    # -----------------------------

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ]
    )

    # -----------------------------
    # Train Linear Regression
    # -----------------------------

    model.fit(
        X_train,
        y_train
    )

    # -----------------------------
    # Prediction
    # -----------------------------

    y_pred = model.predict(X_test)

    comparison = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })

    comparison["Error"] = (
        comparison["Actual"] - comparison["Predicted"]
    )

    comparison["Absolute_Error"] = (
        comparison["Error"].abs()
    )

    print("\n========== SAMPLE PREDICTIONS ==========")
    print(comparison.head(10))

    # -----------------------------
    # Linear Regression Evaluation
    # -----------------------------

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = mean_squared_error(
        y_test,
        y_pred
    ) ** 0.5

    r2 = r2_score(
        y_test,
        y_pred
    )

    print("\n========== LINEAR REGRESSION ==========")

    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R²   : {r2:.4f}")

        # -----------------------------
    # Ridge Regression + GridSearch
    # -----------------------------

    print("\n========== RIDGE HYPERPARAMETER TUNING ==========")

    ridge_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)[0]),
            ("regressor", Ridge())
        ]
    )

    param_grid = {
        "regressor__alpha": [
            0.01,
            0.1,
            1,
            10,
            50,
            100,
            500,
            1000
        ]
    }

    grid_search = GridSearchCV(
        estimator=ridge_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Ridge Parameters:")
    print(grid_search.best_params_)

    print("\nBest CV RMSE:")
    print(-grid_search.best_score_)

    # -----------------------------
    # Best Ridge Model
    # -----------------------------

    best_ridge = grid_search.best_estimator_

    y_pred_ridge = best_ridge.predict(X_test)

    mae_ridge = mean_absolute_error(
        y_test,
        y_pred_ridge
    )

    rmse_ridge = mean_squared_error(
        y_test,
        y_pred_ridge
    ) ** 0.5

    r2_ridge = r2_score(
        y_test,
        y_pred_ridge
    )

    print("\n========== TUNED RIDGE REGRESSION ==========")

    print(f"MAE  : {mae_ridge:,.2f}")
    print(f"RMSE : {rmse_ridge:,.2f}")
    print(f"R²   : {r2_ridge:.4f}")


    rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor(X_train)[0]),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

if __name__ == "__main__":
    train_model()