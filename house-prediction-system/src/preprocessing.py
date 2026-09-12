import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(X: pd.DataFrame):

    # MS SubClass is numeric in the dataset,
    # but semantically it represents a category.
    categorical_features = X.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    if "MS SubClass" in X.columns:
        categorical_features.append("MS SubClass")

    numerical_features = [
        col for col in X.columns
        if col not in categorical_features
    ]

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return (
        preprocessor,
        numerical_features,
        categorical_features
    )