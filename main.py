from utility import DataPreprocessor
import settings
from plotting import DataPlotter
from chronos_predictor import ChronosPredictor


def main():
    preprocessor = DataPreprocessor(settings.FILE_PATH)
    chronos_predictor = ChronosPredictor()
    plotter = DataPlotter(output_folder='output')

    df = preprocessor.load_and_preprocess_data()
    df_example = preprocessor.load_and_preprocess_from_url()

    train_data, test_data = preprocessor.train_test_split(df=df_example, prediction_length=settings.PREDICTION_LENGTH)
    # plotter.plot_time_series(train_data=train_data, test_data=test_data, file_name='train_test_data_plot.png')
    predictor = chronos_predictor.fit_predictor(prediction_length=settings.PREDICTION_LENGTH, train_data=train_data)
    print("Predictor fitted successfully")


    print(df["target"].dtype)


if __name__ == "__main__":
    main()
