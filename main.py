from utility import DataPreprocessor


def main():
    file_path = 'input/medium_views_published_holidays.csv'

    preprocessor = DataPreprocessor(file_path)

    df = preprocessor.load_and_preprocess_data()


    print(df.head())


if __name__ == "__main__":
    main()

