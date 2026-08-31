"""
Tests for the FedMed Phase 3.1 federated parameter contract.

Covers:

- Parameter contract construction
- State-dict ordering
- Parameter extraction
- Defensive copying
- Parameter validation
- Shape/count/dtype validation
- Non-finite-value rejection
- Parameter loading
- Model round-trip behavior
- Contract compatibility
- Invalid model/payload handling
"""

import numpy as np
import pytest
import torch
from torch import nn

from src.common.exceptions import FederatedLearningError
from src.fl.parameters import (
    ParameterContract,
    ParameterContractError,
    copy_parameters,
    extract_parameters,
    load_parameters,
    validate_contract_compatibility,
    validate_parameters,
)
from src.models.base_model import BaseModel


class DummyParameterModel(BaseModel):
    """Minimal model used exclusively for parameter-contract tests."""

    def build(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )


class DifferentParameterModel(BaseModel):
    """Model with a deliberately incompatible state layout."""

    def build(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(4, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
        )


@pytest.fixture
def model() -> DummyParameterModel:
    """Create a fresh test model."""

    torch.manual_seed(42)

    return DummyParameterModel(
        name="parameter_contract_model",
        device="cpu",
    )


# ============================================================
# Contract construction
# ============================================================


def test_parameter_contract_is_federated_learning_error() -> None:
    """Verify the contract exception belongs to the FL hierarchy."""

    assert issubclass(
        ParameterContractError,
        FederatedLearningError,
    )


def test_contract_from_model(model: DummyParameterModel) -> None:
    """Verify a contract is constructed from the model state."""

    contract = ParameterContract.from_model(model)

    assert contract.count == len(model.state_dict())
    assert len(contract.names) == contract.count
    assert len(contract.shapes) == contract.count


def test_contract_preserves_state_dict_order(
    model: DummyParameterModel,
) -> None:
    """Verify contract ordering exactly follows state_dict ordering."""

    contract = ParameterContract.from_model(model)

    expected_names = tuple(model.state_dict().keys())

    assert contract.names == expected_names


def test_contract_shapes_match_model(
    model: DummyParameterModel,
) -> None:
    """Verify recorded shapes match the model state."""

    contract = ParameterContract.from_model(model)

    expected_shapes = tuple(
        tuple(tensor.shape)
        for tensor in model.state_dict().values()
    )

    assert contract.shapes == expected_shapes


# ============================================================
# Extraction
# ============================================================


def test_extract_parameters_returns_numpy_arrays(
    model: DummyParameterModel,
) -> None:
    """Verify extraction returns a NumPy-array payload."""

    parameters = extract_parameters(model)

    assert isinstance(parameters, list)
    assert len(parameters) == len(model.state_dict())

    assert all(
        isinstance(parameter, np.ndarray)
        for parameter in parameters
    )


def test_extract_parameters_matches_state_dict(
    model: DummyParameterModel,
) -> None:
    """Verify extracted arrays match the model state values."""

    parameters = extract_parameters(model)

    for parameter, tensor in zip(
        parameters,
        model.state_dict().values(),
    ):
        np.testing.assert_array_equal(
            parameter,
            tensor.detach().cpu().numpy(),
        )


def test_extract_parameters_is_defensive_copy(
    model: DummyParameterModel,
) -> None:
    """
    Verify modifying an extracted payload does not modify
    the underlying model.
    """

    parameters = extract_parameters(model)

    original = model.state_dict()

    before = {
        key: value.clone()
        for key, value in original.items()
    }

    parameters[0].flat[0] += 1000.0

    after = model.state_dict()

    for key in before:
        torch.testing.assert_close(
            before[key],
            after[key],
        )


# ============================================================
# Validation
# ============================================================


def test_validate_valid_parameters(
    model: DummyParameterModel,
) -> None:
    """Verify a valid payload passes validation."""

    contract = ParameterContract.from_model(model)
    parameters = extract_parameters(model)

    validate_parameters(parameters, contract)


def test_validate_rejects_parameter_count_mismatch(
    model: DummyParameterModel,
) -> None:
    """Verify missing parameters are rejected."""

    contract = ParameterContract.from_model(model)
    parameters = extract_parameters(model)

    parameters.pop()

    with pytest.raises(ParameterContractError):
        validate_parameters(parameters, contract)


def test_validate_rejects_non_numpy_parameter(
    model: DummyParameterModel,
) -> None:
    """Verify non-NumPy parameter values are rejected."""

    contract = ParameterContract.from_model(model)
    parameters = extract_parameters(model)

    parameters[0] = parameters[0].tolist()

    with pytest.raises(ParameterContractError):
        validate_parameters(parameters, contract)


def test_validate_rejects_shape_mismatch(
    model: DummyParameterModel,
) -> None:
    """Verify incorrect parameter shapes are rejected."""

    contract = ParameterContract.from_model(model)
    parameters = extract_parameters(model)

    parameters[0] = np.zeros(
        (999, 999),
        dtype=parameters[0].dtype,
    )

    with pytest.raises(ParameterContractError):
        validate_parameters(parameters, contract)


def test_validate_rejects_dtype_mismatch(
    model: DummyParameterModel,
) -> None:
    """Verify incompatible NumPy dtypes are rejected."""

    contract = ParameterContract.from_model(model)
    parameters = extract_parameters(model)

    parameters[0] = parameters[0].astype(np.float64)

    with pytest.raises(ParameterContractError):
        validate_parameters(parameters, contract)


def test_validate_rejects_nan(
    model: DummyParameterModel,
) -> None:
    """Verify NaN values are rejected."""

    contract = ParameterContract.from_model(model)
    parameters = extract_parameters(model)

    parameters[0].flat[0] = np.nan

    with pytest.raises(ParameterContractError):
        validate_parameters(parameters, contract)


def test_validate_rejects_positive_infinity(
    model: DummyParameterModel,
) -> None:
    """Verify positive infinity is rejected."""

    contract = ParameterContract.from_model(model)
    parameters = extract_parameters(model)

    parameters[0].flat[0] = np.inf

    with pytest.raises(ParameterContractError):
        validate_parameters(parameters, contract)


def test_validate_rejects_negative_infinity(
    model: DummyParameterModel,
) -> None:
    """Verify negative infinity is rejected."""

    contract = ParameterContract.from_model(model)
    parameters = extract_parameters(model)

    parameters[0].flat[0] = -np.inf

    with pytest.raises(ParameterContractError):
        validate_parameters(parameters, contract)


# ============================================================
# Copy behavior
# ============================================================


def test_copy_parameters_returns_independent_arrays(
    model: DummyParameterModel,
) -> None:
    """Verify copied payloads do not share array memory."""

    original = extract_parameters(model)
    copied = copy_parameters(original)

    assert copied is not original
    assert len(copied) == len(original)

    for source, destination in zip(original, copied):
        assert source is not destination
        assert not np.shares_memory(source, destination)

        np.testing.assert_array_equal(
            source,
            destination,
        )


def test_copy_parameters_rejects_non_numpy_values() -> None:
    """Verify copying rejects invalid payload values."""

    with pytest.raises(ParameterContractError):
        copy_parameters(
            [
                np.zeros(2, dtype=np.float32),
                [1.0, 2.0],
            ]
        )


# ============================================================
# Loading
# ============================================================


def test_load_parameters_restores_model(
    model: DummyParameterModel,
) -> None:
    """Verify loading restores the original model state."""

    original = extract_parameters(model)

    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.add_(10.0)

    load_parameters(model, original)

    restored = extract_parameters(model)

    for expected, actual in zip(original, restored):
        np.testing.assert_array_equal(
            expected,
            actual,
        )


def test_load_parameters_rejects_invalid_payload(
    model: DummyParameterModel,
) -> None:
    """Verify invalid payloads are rejected before loading."""

    parameters = extract_parameters(model)
    parameters.pop()

    with pytest.raises(ParameterContractError):
        load_parameters(model, parameters)


def test_load_parameters_does_not_retain_input_reference(
    model: DummyParameterModel,
) -> None:
    """
    Verify changing the caller's payload after loading does not
    modify the loaded model.
    """

    parameters = extract_parameters(model)

    expected = [parameter.copy() for parameter in parameters]

    load_parameters(model, parameters)

    parameters[0].fill(999.0)

    actual = extract_parameters(model)

    for expected_array, actual_array in zip(
        expected,
        actual,
    ):
        np.testing.assert_array_equal(
            expected_array,
            actual_array,
        )


# ============================================================
# Round-trip behavior
# ============================================================


def test_parameter_round_trip(
    model: DummyParameterModel,
) -> None:
    """
    Verify model → federated payload → model produces identical
    state.
    """

    original = extract_parameters(model)

    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.normal_()

    load_parameters(model, original)

    restored = extract_parameters(model)

    for expected, actual in zip(original, restored):
        np.testing.assert_array_equal(
            expected,
            actual,
        )


# ============================================================
# Contract compatibility
# ============================================================


def test_compatible_contracts_pass(
    model: DummyParameterModel,
) -> None:
    """Verify identical model layouts are compatible."""

    first = ParameterContract.from_model(model)
    second = ParameterContract.from_model(model)

    validate_contract_compatibility(first, second)


def test_incompatible_model_contracts_are_rejected(
    model: DummyParameterModel,
) -> None:
    """Verify structurally different models are incompatible."""

    other_model = DifferentParameterModel(
        name="different_parameter_model",
        device="cpu",
    )

    first = ParameterContract.from_model(model)
    second = ParameterContract.from_model(other_model)

    with pytest.raises(ParameterContractError):
        validate_contract_compatibility(first, second)


# ============================================================
# Invalid model handling
# ============================================================


def test_extract_parameters_rejects_non_model() -> None:
    """Verify extraction rejects arbitrary objects."""

    with pytest.raises(ParameterContractError):
        extract_parameters(object())  # type: ignore[arg-type]


def test_contract_rejects_non_model() -> None:
    """Verify contract construction rejects arbitrary objects."""

    with pytest.raises(ParameterContractError):
        ParameterContract.from_model(object())  # type: ignore[arg-type]


def test_load_parameters_rejects_non_model() -> None:
    """Verify loading rejects arbitrary objects."""

    with pytest.raises(ParameterContractError):
        load_parameters(
            object(),  # type: ignore[arg-type]
            [],
        )