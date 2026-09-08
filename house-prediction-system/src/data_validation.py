from data_ingestion import DataIngestion



def validate_data(df):

    print("\n========== DATA VALIDATION ==========")

    # 1. Check whether dataframe is empty
    if df.empty:
        print("❌ Dataset is empty")
        return False

    print("✅ Dataset is not empty")

    # 2. Check target column
    if "SalePrice" not in df.columns:
        print("❌ SalePrice column is missing")
        return False

    print("✅ SalePrice column exists")

    # 3. Check number of rows and columns
    print(f"\nRows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # 4. Check duplicate rows
    duplicates = df.duplicated().sum()

    print(f"\nDuplicate rows: {duplicates}")

    # 5. Check missing values
    missing_values = df.isnull().sum()

    missing_values = missing_values[missing_values > 0
    ].sort_values(ascending=False)

    print("\nMissing values:")
    print(missing_values)

    # 6. Check target statistics
    print("\nTarget statistics:")
    print(df["SalePrice"].describe())

    print("\n========== VALIDATION COMPLETE ==========")

    return True


if __name__ == "__main__":

    ingestion = DataIngestion(
        "data/raw/AmesHousing.csv"
    )

    df = ingestion.load_data()

    validate_data(df)