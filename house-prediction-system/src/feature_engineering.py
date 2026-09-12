# Why are we doing this?
# Your notebook was the research environment.
# Now we're converting the research into reusable production code.

import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Remove identification columns
    df = df.drop(
        columns=["Order", "PID"],
        errors="ignore"
    )

    # -------------------------
    # Aggregate Features
    # -------------------------

    df["TotalSF"] = (
        df["Total Bsmt SF"]
        + df["1st Flr SF"]
        + df["2nd Flr SF"]
    )

    df["TotalBathrooms"] = (
        df["Full Bath"]
        + 0.5 * df["Half Bath"]
        + df["Bsmt Full Bath"]
        + 0.5 * df["Bsmt Half Bath"]
    )

    df["TotalPorchSF"] = (
        df["Wood Deck SF"]
        + df["Open Porch SF"]
        + df["Enclosed Porch"]
        + df["3Ssn Porch"]
        + df["Screen Porch"]
    )

    df["TotalBsmtFinSF"] = (
        df["BsmtFin SF 1"]
        + df["BsmtFin SF 2"]
    )

    # -------------------------
    # Age Features
    # -------------------------

    df["HouseAge"] = (
        df["Yr Sold"] - df["Year Built"]
    ).clip(lower=0)

    df["YearsSinceRemodel"] = (
        df["Yr Sold"] - df["Year Remod/Add"]
    ).clip(lower=0)

    df["GarageAge"] = (
        df["Yr Sold"] - df["Garage Yr Blt"]
    ).clip(lower=0)

    # -------------------------
    # Log Features
    # -------------------------

    log_features = [
        "Lot Area",
        "TotalSF",
        "Gr Liv Area",
        "Total Bsmt SF",
        "1st Flr SF",
        "BsmtFin SF 1",
        "TotalPorchSF"
    ]

    for col in log_features:

        df[f"Log_{col.replace(' ', '_')}"] = (
            np.log1p(df[col])
        )

    return df