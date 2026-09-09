import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(X):

    # -------------------------
    # Identify numerical and categorical features
    # -------------------------   
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_features = X.select_dtypes(include=["object", "str"]).columns.tolist()

    # -------------------------
    # Numerical pipeline
    # -------------------------

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore",sparse_output=False))
        ]
    )

    # -------------------------
    # Combine pipelines using ColumnTransformer
    # -------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor,numerical_features, categorical_features

# print("\nNumber of numerical features:", len(numerical_features))
# print("Numerical features:")
# print(numerical_features)

# print("\nNumber of categorical features", len(categorical_features))
# print("Categorical features:")
# print(categorical_features)

# print("\n========== Missing Values ==========")

# print("\nNumerical features with missing values:")

# numerical_missing = X[numerical_features].isnull().sum()

# print(
#     numerical_missing[numerical_missing > 0]
#     .sort_values(ascending=False)
# )


# print("\nCategorical features with missing values:")

# categorical_missing = X[categorical_features].isnull().sum()

# print(
#     categorical_missing[categorical_missing > 0]
#     .sort_values(ascending=False)
# )

if __name__ == "__main__":

    df = pd.read_csv("data/raw/AmesHousing.csv")

    X = df.drop(columns=["SalePrice"])

    preprocessor = built_preprocessor(X)

    X_transformed = preprocessor.fit_transform(X)

    print("Original shape:")
    print(X.shape)

    print("\nTransformed shape:")
    print(X_transformed.shape)

    print("\nMissing values after preprocessing:")
    print(pd.DataFrame(X_transformed).isna().sum().sum())

    print("\n========== PREPROCESSING COMPLETE ==========")