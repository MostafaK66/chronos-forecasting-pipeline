from autogluon.timeseries import TimeSeriesPredictor


class ChronosPredictor:
    def __init__(self):
        self.predictor = None

    def fit_predictor(self, prediction_length, train_data, model_size):
        self.predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
            train_data, hyperparameters={
        "Chronos": {
            "model_path": "tiny",
            "batch_size": 16,
            "device": "cpu",
        }
    },  presets=model_size, refit_full=False

        )
        return self.predictor
