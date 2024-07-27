import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame


class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.file_path)
        df = df.rename(
            columns={"unique_id": "item_id", "ds": "timestamp", "y": "target"}
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df = df.drop(columns=["published", "is_holiday"], errors="ignore")

        df["target"] = df["target"].astype("float64")

        required_columns = ["item_id", "timestamp", "target"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df = df.set_index(["item_id", "timestamp"])

        df = TimeSeriesDataFrame(df)
        return df

    def train_test_split(self, df, prediction_length):
        train_data, test_data = df.train_test_split(prediction_length=prediction_length)
        return train_data, test_data
