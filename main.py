import settings
from chronos_predictor import ChronosPredictor
from plotting import DataPlotter
from utility import DataPreprocessor


def main():
    preprocessor = DataPreprocessor(settings.FILE_PATH)
    chronos_predictor = ChronosPredictor()
    plotter = DataPlotter(output_folder="output")

    df = preprocessor.load_and_preprocess_data()

    train_data, test_data = preprocessor.train_test_split(
        df=df, prediction_length=settings.PREDICTION_LENGTH
    )
    plotter.plot_time_series(
        train_data=train_data, test_data=test_data, file_name="train_test_data_plot.png"
    )
    predictor = chronos_predictor.fit_predictor(
        prediction_length=settings.PREDICTION_LENGTH,
        train_data=train_data,
        model_size=settings.MODEL_SIZE,
        custom_parameters=settings.CUSTOM_HYPERPARAMETERS,
        hyperparameter_tuning_type=settings.HYPERPARAMETER_TUNING_TYPES,
    )
    print("Predictor fitted successfully")

    print(df["target"].dtype)


if __name__ == "__main__":
    main()
