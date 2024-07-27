from autogluon.timeseries import TimeSeriesPredictor


class ChronosPredictor:
    def __init__(self):
        self.predictor = None

    def fit_predictor(
        self,
        prediction_length,
        train_data,
        model_size,
        custom_parameters,
        hyperparameter_tuning_type,
    ):
        self.predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
            train_data,
            hyperparameters=custom_parameters,
            presets=model_size,
            refit_full=False,
            skip_model_selection=False,
            hyperparameter_tune_kwargs=hyperparameter_tuning_type,
        )
        return self.predictor
