from data_ingestion import DataIngestion
import matplotlib.pyplot as plt

def explore_data(df):
    print("\n========== Dataset Overview ==========")

    print("\nData Types:")
    print(df.dtypes)

    print("\nNumerical Features:")
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns

    print(list(numerical_features))

    print("\nCategorical Features:")
    categorical_features = df.select_dtypes(include=["str", "object", "category"]).columns

    print(list(categorical_features))

    print("\nNumber of Numerical Features:")
    print(len(numerical_features))

    print("\nNumber of Categorical Features:")
    print(len(categorical_features))

    print("\nUnique Values:")
    print(df.nunique().sort_values())

    print("\nSalePrice Skewness:")
    print(df["SalePrice"].skew())

    print("\nSalePrice Quantiles:")
    print(df["SalePrice"].quantile([0.01,0.05,0.25,0.50,0.75,0.95,0.99]))

    plt.figure(figsize=(10,6))

    plt.hist(df["SalePrice"], bins=50)
    plt.xlabel("SalePrice")
    plt.ylabel("Frequency")
    plt.title("Distribution of SalePrice")
    plt.show()

    print("\n========== EDA Complete ==========")

if __name__ == "__main__":

        ingestion = DataIngestion("data/raw/AmesHousing.csv")

        df = ingestion.load_data()
        explore_data(df)