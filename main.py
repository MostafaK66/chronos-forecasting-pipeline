from utility import DataPreprocessor
import settings
from plotting import DataPlotter


def main():
    preprocessor = DataPreprocessor(settings.FILE_PATH)
    plotter = DataPlotter(output_folder='output')

    df = preprocessor.load_and_preprocess_data()
    plotter.plot_time_series(df)

    print(df.index.get_level_values('item_id'))
    print(df.index.get_level_values('timestamp'))
    print(df.head())


if __name__ == "__main__":
    main()
