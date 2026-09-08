import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit 


def evaluate_model(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true,y_pred)

    directional_accuracy = np.mean(
        np.sign(y_true) == np.sign(y_pred)
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy": directional_accuracy
    }


if __name__ == "__main__":

    from dataset import create_dataset, time_series_split

    X, y, df = create_dataset("AAPL")

    X_train, X_test, y_train, y_test = time_series_split(
        X,
        y
    )

    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        "n_estimators": [100,200,300],
        "max_depth": [3,5,8,None],
        "min_samples_split": [2,5,10],
        "min_samples_leaf": [2,5,10],
        "max_features": [0.5, 1.0,None]
    }

    # -------------------------
    # Baseline
    # -------------------------

    y_pred_baseline = X_test["return_1d"]

    metrics_baseline = evaluate_model(
        y_test,
        y_pred_baseline
    )

    print("\nBaseline Performance")

    
    for metric, value in metrics_baseline.items():
        print(f"{metric}: {value:.4f}")

    # -------------------------
    # Linear Regression
    # -------------------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)

    print("\nActual vs Predicted Returns (Linear Regression):")
    comparison = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred_lr
    })
    print(comparison.head(10))

    print("\\n LR Coefficient:")
    print(lr.coef_)
    print("\n LR Intercept")
    print(lr.intercept_)

    metrics_lr = evaluate_model(y_test,y_pred_lr)
    print("\nLinear Regression Performance")


    for metric, value in metrics_lr.items():
        print(f"{metric}: {value:.4f}")

    # -------------------------
    # Random Forest
    # -------------------------

    rf = RandomForestRegressor(n_estimators=300,
                               max_depth=5,
                               min_samples_leaf=10,
                               random_state=42,
                               n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    ) 
    grid_search.fit(X_train, y_train)

    # rf.fit(X_train,y_train)
    # y_pred_rf = rf.predict(X_test)

    # print("\nActual vs Predicted Returns (Random Forest):")
    # comparison_rf = pd.DataFrame({
    #     "Actual": y_test,
    #     "Predicted": y_pred_rf
    # })
    # print(comparison_rf.head(10))

    
    # metrics_rf = evaluate_model(y_test,y_pred_rf)
    # print("\nRandom Forest Performance")

    # for metrics, value in metrics_rf.items():
    #     print(f"{metrics}: {value:.4f}")

    print("\nBest Random Forest Parameters:")
    print(grid_search.best_params_)

    print("\nBest CV score:")
    print(-grid_search.best_score_)

    best_rf = grid_search.best_estimator_

    y_pred_rf = best_rf.predict(X_test)

    metrics_rf = evaluate_model(
        y_test,
        y_pred_rf
    )

    print("\nTuned Random Forest Performance")

    for metric, value in metrics_rf.items():
        print(f"{metric}: {value:.4f}")

    from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

metrics_xgb = evaluate_model(y_test, y_pred_xgb)

print("\nXGBoost Performance")

for metric, value in metrics_xgb.items():
    print(f"{metric}: {value:.4f}")

# on comparing the performance of XGBoost with Random Forest
print("\nComparison of XGBoost and Random Forest Performance")
print("Accuracy of XGBoost: {:.4f}".format(metrics_xgb["Directional Accuracy"]))
print("Accuracy of Random Forest: {:.4f}".format(metrics_rf["Directional Accuracy"]))