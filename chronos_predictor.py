from autogluon.timeseries import TimeSeriesPredictor

import settings


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
            num_val_windows=settings.NUM_VALIDATION_WINDOW,
            refit_every_n_windows=settings.REFIT_EVERY_N_WINDOWS,
        )
        return self.predictor
