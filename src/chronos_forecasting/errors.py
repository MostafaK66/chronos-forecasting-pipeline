"""Expected application failures."""


class ForecastingError(Exception):
    """Base class for recoverable forecasting errors."""


class ConfigurationError(ForecastingError):
    """Configuration is invalid or cannot be loaded."""


class DataValidationError(ForecastingError):
    """Input time-series data violates the schema or history contract."""


class DependencyUnavailableError(ForecastingError):
    """An optional runtime dependency is not installed."""


class BackendError(ForecastingError):
    """The forecasting backend failed or returned malformed output."""


class ArtifactError(ForecastingError):
    """A generated artifact could not be validated or written."""
