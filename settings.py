from autogluon.common import space

FILE_PATH = "input/medium_views_published_holidays.csv"
PREDICTION_LENGTH = 100
MODEL_SIZE = "chronos_tiny"

CUSTOM_HYPERPARAMETERS = {
    "Chronos": {
        "model_path": "tiny",
        "batch_size": space.Int(8, 16),
        "learning_rate": space.Categorical(1e-3, 1e-4),
    }
}
