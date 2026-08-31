class FedMedError(Exception):
    """
    Base exception for the FedMed application.

    All custom FedMed exceptions inherit from this class.
    This allows application-level code to catch any FedMed-specific
    error using a single exception type.
    """

    pass


class ConfigurationError(FedMedError):
    """
    Raised when application configuration is missing,
    invalid, or cannot be loaded.
    """

    pass


class ModelError(FedMedError):
    """
    Raised when model creation, loading, parameter handling,
    or model-related operations fail.
    """

    pass


class DataError(FedMedError):
    """
    Raised when dataset loading, validation, partitioning,
    or data-related operations fail.
    """

    pass


class TrainingError(FedMedError):
    """
    Raised when local model training or evaluation fails.
    """

    pass


class FederatedLearningError(FedMedError):
    """
    Base exception for federated-learning related failures.
    """

    pass


class AggregationError(FederatedLearningError):
    """
    Raised when client model updates cannot be aggregated
    into a global model.
    """

    pass