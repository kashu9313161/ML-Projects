from pathlib import Path
import pandas as pd


# Why a class? => Because later our ingestion component may have more responsibilities:
class DataIngestion:

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_data(self) -> pd.DataFrame:

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}"
            )

        df = pd.read_csv(self.data_path)
        return df

if __name__ == "__main__":

    ingestion = DataIngestion("data/raw/AmesHousing.csv")

    df = ingestion.load_data()

    print("Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())