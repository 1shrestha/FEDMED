"""Validation tests for the centralized configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.config import ConfigLoader, load_config
from src.common.exceptions import ConfigurationError


VALID_CONFIG = {
    "application": {
        "name": "fedmed",
        "environment": "development",
        "version": "0.1.0",
    },
    "federated": {
        "strategy": "fedavg",
        "num_rounds": 3,
        "min_clients": 1,
        "min_available_clients": 1,
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
    },
    "training": {
        "local_epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.01,
        "optimizer": "sgd",
        "seed": 42,
    },
    "model": {
        "name": "flower_smoke_test_model",
        "device": "cpu",
    },
    "data": {
        "root_dir": "./data",
        "num_clients": 2,
        "partition_type": "iid",
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8080,
    },
    "checkpoint": {
        "enabled": False,
        "directory": "./checkpoints",
        "save_every_round": 1,
    },
    "logging": {
        "level": "INFO",
        "directory": "./logs",
    },
}


@pytest.fixture
def temp_yaml_file(tmp_path: Path) -> Path:
    """Build a valid temporary config file."""
    path = tmp_path / "config.yaml"
    path.write_text(
        """
application:
  name: fedmed
  environment: development
  version: "0.1.0"

federated:
  strategy: fedavg
  num_rounds: 3
  min_clients: 1
  min_available_clients: 1
  fraction_fit: 1.0
  fraction_evaluate: 1.0

training:
  local_epochs: 1
  batch_size: 4
  learning_rate: 0.01
  optimizer: sgd
  seed: 42

model:
  name: flower_smoke_test_model
  device: cpu

data:
  root_dir: ./data
  num_clients: 2
  partition_type: iid

server:
  host: 127.0.0.1
  port: 8080

checkpoint:
  enabled: false
  directory: ./checkpoints
  save_every_round: 1

logging:
  level: INFO
  directory: ./logs
        """.strip(),
        encoding="utf-8",
    )
    return path


def test_load_config_valid_file(temp_yaml_file: Path) -> None:
    """A valid YAML file should produce a typed config object."""
    config = ConfigLoader(temp_yaml_file).load()

    assert config.application.name == "fedmed"
    assert config.federated.num_rounds == 3
    assert config.training.local_epochs == 1
    assert config.model.device == "cpu"
    assert config.server.port == 8080


def test_load_config_missing_file(tmp_path: Path) -> None:
    """A missing file should raise a configuration error."""
    missing = tmp_path / "missing.yaml"

    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        ConfigLoader(missing).load()


def test_load_config_missing_section(temp_yaml_file: Path) -> None:
    """Missing sections should be rejected even if the file exists."""
    temp_yaml_file.write_text(
        """
application:
  name: fedmed
  environment: development
  version: "0.1.0"

training:
  local_epochs: 1
  batch_size: 4
  learning_rate: 0.01
  optimizer: sgd
  seed: 42

model:
  name: smoke
  device: cpu

data:
  root_dir: ./data
  num_clients: 2
  partition_type: iid

server:
  host: 127.0.0.1
  port: 8080

checkpoint:
  enabled: false
  directory: ./checkpoints
  save_every_round: 1

logging:
  level: INFO
  directory: ./logs
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="Missing configuration sections"):
        ConfigLoader(temp_yaml_file).load()


def test_load_config_malformed_yaml(tmp_path: Path) -> None:
    """Malformed YAML should be rejected."""
    bad_file = tmp_path / "broken.yaml"
    bad_file.write_text("training: [1, 2,\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid YAML configuration"):
        ConfigLoader(bad_file).load()


def test_load_config_empty_file(tmp_path: Path) -> None:
    """An empty file should be rejected."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Configuration file is empty"):
        ConfigLoader(empty).load()


def test_load_config_wrong_root_type(tmp_path: Path) -> None:
    """The YAML root must be a mapping."""
    path = tmp_path / "root-list.yaml"
    path.write_text("- item1\n- item2\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Root configuration must be a YAML mapping"):
        ConfigLoader(path).load()


def test_load_config_type_invalid_field_values(temp_yaml_file: Path) -> None:
    """A section with the wrong YAML type should fail during config binding."""
    temp_yaml_file.write_text(
        """
application: "not-a-mapping"

federated:
  strategy: fedavg
  num_rounds: 3
  min_clients: 1
  min_available_clients: 1
  fraction_fit: 1.0
  fraction_evaluate: 1.0

training:
  local_epochs: 1
  batch_size: 4
  learning_rate: 0.01
  optimizer: sgd
  seed: 42

model:
  name: smoke
  device: cpu

data:
  root_dir: ./data
  num_clients: 2
  partition_type: iid

server:
  host: 127.0.0.1
  port: 8080

checkpoint:
  enabled: false
  directory: ./checkpoints
  save_every_round: 1

logging:
  level: INFO
  directory: ./logs
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="Invalid configuration values"):
        ConfigLoader(temp_yaml_file).load()


def test_load_config_helper_uses_default_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The convenience loader should resolve the repo-relative default config path."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        """
application:
  name: fedmed
  environment: development
  version: "0.1.0"

federated:
  strategy: fedavg
  num_rounds: 3
  min_clients: 1
  min_available_clients: 1
  fraction_fit: 1.0
  fraction_evaluate: 1.0

training:
  local_epochs: 1
  batch_size: 4
  learning_rate: 0.01
  optimizer: sgd
  seed: 42

model:
  name: fedmed_model
  device: cpu

data:
  root_dir: ./data
  num_clients: 2
  partition_type: iid

server:
  host: 127.0.0.1
  port: 8080

checkpoint:
  enabled: false
  directory: ./checkpoints
  save_every_round: 1

logging:
  level: INFO
  directory: ./logs
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    config = load_config()

    assert config.application.name == "fedmed"
    assert config.training.learning_rate == 0.01
