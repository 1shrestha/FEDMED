"""
Tests for the FedMed model foundation.

This module tests:

- BaseModel construction
- Forward pass
- Training/evaluation modes
- Parameter extraction
- Parameter restoration
- Parameter validation
- Device handling
- Model metadata
- ModelFactory registration
- ModelFactory creation
- ModelFactory validation
"""

import numpy as np
import pytest
import torch
from torch import nn

from src.common.exceptions import ModelError
from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory


class DummyModel(BaseModel):
    """
    Minimal PyTorch model used exclusively for testing.

    This is intentionally not part of the production model layer.
    """

    def build(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
        )


@pytest.fixture(autouse=True)
def reset_model_factory() -> None:
    """
    Reset the factory registry before every test.

    This prevents tests from affecting one another.
    """

    ModelFactory.clear_registry()


@pytest.fixture
def model() -> DummyModel:
    """Create a fresh test model."""

    return DummyModel(
        name="test_model",
        device="cpu",
    )


# ============================================================
# BaseModel tests
# ============================================================


def test_model_builds_successfully(model: DummyModel) -> None:
    """Verify that a concrete BaseModel can be created."""

    assert model.name == "test_model"
    assert isinstance(model.network, nn.Module)
    assert model.device == torch.device("cpu")


def test_forward_pass(model: DummyModel) -> None:
    """Verify that the model produces the expected output shape."""

    inputs = torch.randn(4, 10)

    outputs = model.forward(inputs)

    assert outputs.shape == (4, 2)


def test_train_mode(model: DummyModel) -> None:
    """Verify that the model can enter training mode."""

    model.train_mode()

    assert model.network.training is True


def test_eval_mode(model: DummyModel) -> None:
    """Verify that the model can enter evaluation mode."""

    model.eval_mode()

    assert model.network.training is False


def test_parameter_extraction(model: DummyModel) -> None:
    """Verify that model state can be extracted as NumPy arrays."""

    parameters = model.get_parameters()

    assert isinstance(parameters, list)
    assert len(parameters) > 0

    for parameter in parameters:
        assert isinstance(parameter, np.ndarray)


def test_parameter_round_trip(model: DummyModel) -> None:
    """
    Verify that model parameters can be extracted,
    modified, and restored correctly.
    """

    original_parameters = model.get_parameters()

    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.add_(1.0)

    modified_parameters = model.get_parameters()

    assert any(
        not np.array_equal(original, modified)
        for original, modified in zip(
            original_parameters,
            modified_parameters,
        )
    )

    model.set_parameters(original_parameters)

    restored_parameters = model.get_parameters()

    for original, restored in zip(
        original_parameters,
        restored_parameters,
    ):
        np.testing.assert_array_equal(
            original,
            restored,
        )


def test_parameter_count_mismatch(model: DummyModel) -> None:
    """Verify that an incorrect number of state arrays raises ModelError."""

    parameters = model.get_parameters()

    parameters.pop()

    with pytest.raises(ModelError):
        model.set_parameters(parameters)


def test_parameter_shape_mismatch(model: DummyModel) -> None:
    """Verify that an incorrect tensor shape raises ModelError."""

    parameters = model.get_parameters()

    parameters[0] = np.zeros(
        (999, 999),
        dtype=np.float32,
    )

    with pytest.raises(ModelError):
        model.set_parameters(parameters)


def test_state_dict_round_trip(model: DummyModel) -> None:
    """Verify PyTorch checkpoint state can be saved and restored."""

    original_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
    }

    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.add_(2.0)

    model.load_state_dict(original_state)

    restored_state = model.state_dict()

    for key in original_state:
        torch.testing.assert_close(
            original_state[key],
            restored_state[key],
        )


def test_device_handling(model: DummyModel) -> None:
    """Verify model device management."""

    model.to("cpu")

    assert model.device == torch.device("cpu")

    for parameter in model.network.parameters():
        assert parameter.device == torch.device("cpu")


def test_model_metadata(model: DummyModel) -> None:
    """Verify model metadata."""

    metadata = model.metadata

    assert metadata["name"] == "test_model"
    assert metadata["device"] == "cpu"
    assert metadata["num_parameters"] > 0


# ============================================================
# ModelFactory tests
# ============================================================


def test_register_model() -> None:
    """Verify that a model can be registered."""

    ModelFactory.register(
        "test_model",
        DummyModel,
    )

    assert ModelFactory.is_registered("test_model")


def test_register_normalizes_model_name() -> None:
    """Verify model identifiers are normalized."""

    ModelFactory.register(
        "  TEST_MODEL  ",
        DummyModel,
    )

    assert ModelFactory.is_registered("test_model")


def test_create_registered_model() -> None:
    """Verify that a registered model can be instantiated."""

    ModelFactory.register(
        "test_model",
        DummyModel,
    )

    model = ModelFactory.create(
        "test_model",
        name="factory_model",
        device="cpu",
    )

    assert isinstance(model, DummyModel)
    assert model.name == "factory_model"


def test_available_models() -> None:
    """Verify registered models can be listed."""

    ModelFactory.register(
        "model_b",
        DummyModel,
    )

    ModelFactory.register(
        "model_a",
        DummyModel,
    )

    assert ModelFactory.available_models() == [
        "model_a",
        "model_b",
    ]


def test_unknown_model_raises_error() -> None:
    """Verify that creating an unknown model raises ModelError."""

    with pytest.raises(ModelError):
        ModelFactory.create("unknown_model")


def test_duplicate_registration_raises_error() -> None:
    """Verify duplicate model identifiers are rejected."""

    ModelFactory.register(
        "test_model",
        DummyModel,
    )

    with pytest.raises(ModelError):
        ModelFactory.register(
            "test_model",
            DummyModel,
        )


def test_invalid_model_class_raises_error() -> None:
    """Verify only BaseModel subclasses can be registered."""

    class InvalidModel:
        pass

    with pytest.raises(ModelError):
        ModelFactory.register(
            "invalid",
            InvalidModel,
        )


def test_empty_model_name_raises_error() -> None:
    """Verify empty model identifiers are rejected."""

    with pytest.raises(ModelError):
        ModelFactory.register(
            "",
            DummyModel,
        )


def test_factory_clear_registry() -> None:
    """Verify the registry can be cleared."""

    ModelFactory.register(
        "test_model",
        DummyModel,
    )

    assert ModelFactory.available_models() == ["test_model"]

    ModelFactory.clear_registry()

    assert ModelFactory.available_models() == []