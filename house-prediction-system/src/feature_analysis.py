from data_ingestion import DataIngestion
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_numeric_features(df):

    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns

    correlation = (df[numerical_features].corr()["SalePrice"].sort_values(ascending=False))

    print("\n======= Correlation with SalePrice =======")
    print(correlation)

def plot_top_features(df):

    top_features = ["Overall Qual", "Gr Liv Area", "Garage Cars","Garage Area", "Total Bsmt SF", "1stFlrSF", "Year Built"]

    for feature in top_features:

        plt.figure(figsize=(7,5))

        sns.scatterplot(data=df,x=feature,y="SalePrice")

        plt.title(f"{feature} vs SalePrice")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ingestion = DataIngestion("data/raw/AmesHousing.csv")

    df = ingestion.load_data()
    analyze_numeric_features(df)
    plot_top_features(df)