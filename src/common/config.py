from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.common.exceptions import ConfigurationError


@dataclass(frozen=True)
class ApplicationConfig:
    """General application metadata and runtime environment."""

    name: str
    environment: str
    version: str


@dataclass(frozen=True)
class FederatedConfig:
    """Federated learning configuration."""

    strategy: str
    num_rounds: int
    min_clients: int
    min_available_clients: int
    fraction_fit: float
    fraction_evaluate: float


@dataclass(frozen=True)
class TrainingConfig:
    """Local model training configuration."""

    local_epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    seed: int


@dataclass(frozen=True)
class ModelConfig:
    """Model runtime configuration."""

    name: str
    device: str


@dataclass(frozen=True)
class DataConfig:
    """Dataset and client partitioning configuration."""

    root_dir: str
    num_clients: int
    partition_type: str


@dataclass(frozen=True)
class ServerConfig:
    """Federated server configuration."""

    host: str
    port: int


@dataclass(frozen=True)
class CheckpointConfig:
    """Model checkpoint configuration."""

    enabled: bool
    directory: str
    save_every_round: int


@dataclass(frozen=True)
class LoggingConfig:
    """Application logging configuration."""

    level: str
    directory: str


@dataclass(frozen=True)
class FedMedConfig:
    """
    Complete centralized FedMed configuration.

    All application configuration is represented through this object.
    """

    application: ApplicationConfig
    federated: FederatedConfig
    training: TrainingConfig
    model: ModelConfig
    data: DataConfig
    server: ServerConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig


class ConfigLoader:
    """
    Loads and validates the centralized FedMed configuration.
    """

    REQUIRED_SECTIONS = (
        "application",
        "federated",
        "training",
        "model",
        "data",
        "server",
        "checkpoint",
        "logging",
    )

    SUPPORTED_EXTENSIONS = {".yaml", ".yml"}

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> FedMedConfig:
        """
        Load the YAML configuration and convert it into
        a typed FedMedConfig object.
        """

        self._validate_file()

        raw_config = self._read_yaml()

        self._validate_sections(raw_config)

        return self._build_config(raw_config)

    def _validate_file(self) -> None:
        """Validate that the configuration file exists and is supported."""

        if not self.path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {self.path}"
            )

        if not self.path.is_file():
            raise ConfigurationError(
                f"Configuration path is not a file: {self.path}"
            )

        if self.path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ConfigurationError(
                f"Unsupported configuration format: {self.path.suffix}. "
                f"Expected YAML."
            )

    def _read_yaml(self) -> dict[str, Any]:
        """Read and parse the YAML configuration file."""

        try:
            with self.path.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Invalid YAML configuration: {self.path}"
            ) from exc

        if config is None:
            raise ConfigurationError(
                f"Configuration file is empty: {self.path}"
            )

        if not isinstance(config, dict):
            raise ConfigurationError(
                "Root configuration must be a YAML mapping."
            )

        return config

    def _validate_sections(self, config: dict[str, Any]) -> None:
        """Ensure all required configuration sections exist."""

        missing_sections = [
            section
            for section in self.REQUIRED_SECTIONS
            if section not in config
        ]

        if missing_sections:
            raise ConfigurationError(
                "Missing configuration sections: "
                + ", ".join(missing_sections)
            )

    @staticmethod
    def _build_config(config: dict[str, Any]) -> FedMedConfig:
        """Convert raw configuration data into typed configuration objects."""

        try:
            return FedMedConfig(
                application=ApplicationConfig(
                    **config["application"]
                ),
                federated=FederatedConfig(
                    **config["federated"]
                ),
                training=TrainingConfig(
                    **config["training"]
                ),
                model=ModelConfig(
                    **config["model"]
                ),
                data=DataConfig(
                    **config["data"]
                ),
                server=ServerConfig(
                    **config["server"]
                ),
                checkpoint=CheckpointConfig(
                    **config["checkpoint"]
                ),
                logging=LoggingConfig(
                    **config["logging"]
                ),
            )

        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Invalid configuration values: {exc}"
            ) from exc


def load_config(
    path: str | Path = "configs/config.yaml",
) -> FedMedConfig:
    """
    Load the centralized FedMed configuration.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    FedMedConfig
        Fully parsed and validated configuration.
    """

    return ConfigLoader(path).load()