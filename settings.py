from autogluon.common import space

FILE_PATH = "input/medium_views_published_holidays.csv"
PREDICTION_LENGTH = 100
MODEL_SIZE = "chronos_tiny"

CUSTOM_HYPERPARAMETERS = {
    "Chronos": {
        "model_path": "tiny",
        "batch_size": space.Int(8, 16),
        "learning_rate": space.Categorical(1e-3, 1e-4, 1e-5),
        "num_train_epochs": space.Int(10, 50),
        "dropout_rate": space.Categorical(0.1, 0.2, 0.3),
        "weight_decay": space.Categorical(1e-4, 1e-5, 1e-6),
        "early_stopping_patience": space.Int(5, 10),
        "hidden_size": space.Int(32, 64),
        "num_layers": space.Int(4, 8),
        "optimizer": space.Categorical("adam", "sgd", "adamw"),
    }
}
HYPERPARAMETER_TUNING_TYPES = {"scheduler": "local", "searcher": "auto"}
