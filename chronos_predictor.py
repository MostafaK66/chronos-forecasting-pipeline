from autogluon.timeseries import TimeSeriesPredictor


class ChronosPredictor:
    def __init__(self):
        self.predictor = None

    def fit_predictor(
        self, prediction_length, train_data, model_size, custom_parameters
    ):
        self.predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
            train_data,
            hyperparameters=custom_parameters,
            presets=model_size,
            refit_full=False,
            skip_model_selection=False,
            hyperparameter_tune_kwargs={"scheduler": "local", "searcher": "auto"},
        )
        return self.predictor
