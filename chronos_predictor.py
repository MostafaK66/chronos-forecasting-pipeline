from autogluon.timeseries import TimeSeriesPredictor


class ChronosPredictor:
    def __init__(self):
        self.predictor = None

    def fit_predictor(self, prediction_length, train_data):
        self.predictor = TimeSeriesPredictor(prediction_length).fit(
            train_data, presets="chronos_tiny"

        )
        return self.predictor
