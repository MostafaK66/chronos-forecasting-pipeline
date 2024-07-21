import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame


class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.file_path)
        df = df.rename(columns={'unique_id': 'item_id', 'ds': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = TimeSeriesDataFrame(df)
        return df

