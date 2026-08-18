"""
Federated parameter contract for FedMed.

This module defines the framework-independent parameter boundary
between FedMed models and the federated-learning runtime.

Phase 3.1 responsibilities:

- Define the canonical federated parameter representation.
- Extract model state through BaseModel's existing interface.
- Validate parameter payloads before federated exchange.
- Preserve deterministic state_dict ordering.
- Validate parameter count and shapes.
- Validate NumPy-array types.
- Reject non-finite parameter values.
- Provide defensive copies of parameter payloads.

This module intentionally does NOT:

- implement Flower
- implement client/server communication
- implement aggregation
- implement federated rounds
- modify datasets
- perform local training
- perform evaluation
- contain model-specific architecture logic

Flower-specific conversion belongs to the later federated
runtime layer. FedMed's internal representation remains
``list[np.ndarray]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.common.exceptions import FederatedLearningError
from src.models.base_model import BaseModel


# ============================================================
# Public type aliases
# ============================================================

ParameterPayload = list[np.ndarray]


# ============================================================
# Exceptions
# ============================================================


class ParameterContractError(FederatedLearningError):
    """
    Raised when a federated parameter payload violates the
    FedMed parameter contract.

    This is intentionally more specific than the generic
    FederatedLearningError while remaining inside the existing
    FedMed federated exception hierarchy.
    """

    pass


# ============================================================
# Parameter specification
# ============================================================


@dataclass(frozen=True)
class ParameterSpec:
    """
    Immutable description of one model-state entry.

    Attributes:
        name:
            Exact state_dict key.

        shape:
            Expected tensor shape.

        dtype:
            Expected NumPy dtype representation.

        is_floating_point:
            Whether the corresponding model state is floating-point.
    """

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    is_floating_point: bool


@dataclass(frozen=True)
class ParameterContract:
    """
    Immutable description of a model's federated parameter layout.

    The contract captures the structural information required to
    verify that a parameter payload belongs to the expected model.

    It does not contain the actual parameter values.

    Attributes:
        specs:
            Parameter specifications in exact state_dict order.
    """

    specs: tuple[ParameterSpec, ...]

    @property
    def count(self) -> int:
        """Return the number of state entries in the contract."""

        return len(self.specs)

    @property
    def names(self) -> tuple[str, ...]:
        """Return state_dict names in canonical order."""

        return tuple(spec.name for spec in self.specs)

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        """Return expected shapes in canonical order."""

        return tuple(spec.shape for spec in self.specs)

    @classmethod
    def from_model(cls, model: BaseModel) -> ParameterContract:
        """
        Build a parameter contract from a FedMed BaseModel.

        The contract follows the exact ordering exposed by the
        model's state_dict().

        Args:
            model:
                FedMed model whose parameter layout should be described.

        Returns:
            Immutable ParameterContract.

        Raises:
            ParameterContractError:
                If the model is not a BaseModel.
        """

        if not isinstance(model, BaseModel):
            raise ParameterContractError(
                "ParameterContract.from_model() requires a "
                f"BaseModel, got {type(model).__name__}."
            )

        specs: list[ParameterSpec] = []

        for name, tensor in model.state_dict().items():
            array = tensor.detach().cpu().numpy()

            specs.append(
                ParameterSpec(
                    name=name,
                    shape=tuple(array.shape),
                    dtype=array.dtype,
                    is_floating_point=bool(
                        tensor.is_floating_point()
                    ),
                )
            )

        return cls(specs=tuple(specs))


# ============================================================
# Payload validation
# ============================================================


def validate_parameters(
    parameters: Sequence[np.ndarray],
    contract: ParameterContract,
) -> None:
    """
    Validate a parameter payload against a ParameterContract.

    Validation covers:

    - payload container type
    - parameter count
    - individual NumPy-array type
    - exact parameter shape
    - dtype compatibility
    - finite values for floating-point arrays

    The function does not modify the supplied payload.

    Args:
        parameters:
            Candidate federated parameter payload.

        contract:
            Expected parameter layout.

    Raises:
        ParameterContractError:
            If any part of the payload violates the contract.
    """

    if not isinstance(parameters, (list, tuple)):
        raise ParameterContractError(
            "Federated parameters must be provided as a list or "
            f"tuple of NumPy arrays, got {type(parameters).__name__}."
        )

    if len(parameters) != contract.count:
        raise ParameterContractError(
            "Federated parameter count mismatch: "
            f"expected {contract.count}, received {len(parameters)}."
        )

    for index, (value, spec) in enumerate(
        zip(parameters, contract.specs)
    ):
        if not isinstance(value, np.ndarray):
            raise ParameterContractError(
                f"Federated parameter at index {index} "
                f"('{spec.name}') must be a NumPy array, "
                f"got {type(value).__name__}."
            )

        actual_shape = tuple(value.shape)

        if actual_shape != spec.shape:
            raise ParameterContractError(
                f"Shape mismatch for federated parameter "
                f"'{spec.name}' at index {index}: "
                f"expected {spec.shape}, received {actual_shape}."
            )

        if value.dtype != spec.dtype:
            raise ParameterContractError(
                f"Dtype mismatch for federated parameter "
                f"'{spec.name}' at index {index}: "
                f"expected {spec.dtype}, received {value.dtype}."
            )

        if np.issubdtype(value.dtype, np.inexact):
            if not np.all(np.isfinite(value)):
                raise ParameterContractError(
                    f"Federated parameter '{spec.name}' at index "
                    f"{index} contains non-finite values."
                )


# ============================================================
# Extraction
# ============================================================


def extract_parameters(model: BaseModel) -> ParameterPayload:
    """
    Extract a validated, defensive federated parameter payload.

    This delegates actual model-state extraction to the existing
    BaseModel.get_parameters() boundary, then validates the
    resulting payload against a freshly constructed contract.

    Args:
        model:
            FedMed BaseModel.

    Returns:
        Independent NumPy-array copies representing the complete
        model state in state_dict order.

    Raises:
        ParameterContractError:
            If the model is invalid or produces an invalid payload.
    """

    if not isinstance(model, BaseModel):
        raise ParameterContractError(
            "extract_parameters() requires a BaseModel, "
            f"got {type(model).__name__}."
        )

    contract = ParameterContract.from_model(model)

    try:
        parameters = model.get_parameters()
    except Exception as exc:
        raise ParameterContractError(
            f"Failed to extract parameters from model "
            f"'{model.name}': {exc}"
        ) from exc

    validate_parameters(parameters, contract)

    return [parameter.copy() for parameter in parameters]


# ============================================================
# Defensive copy
# ============================================================


def copy_parameters(
    parameters: Sequence[np.ndarray],
) -> ParameterPayload:
    """
    Create independent copies of a federated parameter payload.

    Args:
        parameters:
            NumPy-array parameter payload.

    Returns:
        A new list containing independent NumPy-array copies.

    Raises:
        ParameterContractError:
            If the payload is not a sequence of NumPy arrays.
    """

    if not isinstance(parameters, (list, tuple)):
        raise ParameterContractError(
            "copy_parameters() requires a list or tuple of NumPy "
            f"arrays, got {type(parameters).__name__}."
        )

    copied: ParameterPayload = []

    for index, parameter in enumerate(parameters):
        if not isinstance(parameter, np.ndarray):
            raise ParameterContractError(
                f"Parameter at index {index} must be a NumPy array, "
                f"got {type(parameter).__name__}."
            )

        copied.append(parameter.copy())

    return copied


# ============================================================
# Loading
# ============================================================


def load_parameters(
    model: BaseModel,
    parameters: Sequence[np.ndarray],
) -> None:
    """
    Validate and load a federated parameter payload into a model.

    The model's existing set_parameters() implementation remains
    responsible for the actual PyTorch state restoration.

    Args:
        model:
            Target FedMed model.

        parameters:
            Candidate federated parameter payload.

    Raises:
        ParameterContractError:
            If the model or payload violates the contract.
    """

    if not isinstance(model, BaseModel):
        raise ParameterContractError(
            "load_parameters() requires a BaseModel, "
            f"got {type(model).__name__}."
        )

    contract = ParameterContract.from_model(model)

    validate_parameters(parameters, contract)

    safe_parameters = copy_parameters(parameters)

    try:
        model.set_parameters(safe_parameters)
    except Exception as exc:
        raise ParameterContractError(
            f"Failed to load federated parameters into model "
            f"'{model.name}': {exc}"
        ) from exc


# ============================================================
# Contract compatibility
# ============================================================


def validate_contract_compatibility(
    source: ParameterContract,
    target: ParameterContract,
) -> None:
    """
    Verify that two parameter contracts describe the same model
    state layout.

    This is useful later when a server receives updates from
    multiple clients.

    Args:
        source:
            Contract describing the incoming update.

        target:
            Contract describing the expected global model.

    Raises:
        ParameterContractError:
            If count, ordering, shapes, or dtypes differ.
    """

    if source.count != target.count:
        raise ParameterContractError(
            "Parameter contract count mismatch: "
            f"source has {source.count}, "
            f"target expects {target.count}."
        )

    for index, (source_spec, target_spec) in enumerate(
        zip(source.specs, target.specs)
    ):
        if source_spec.name != target_spec.name:
            raise ParameterContractError(
                "Parameter ordering/name mismatch at index "
                f"{index}: source='{source_spec.name}', "
                f"target='{target_spec.name}'."
            )

        if source_spec.shape != target_spec.shape:
            raise ParameterContractError(
                f"Parameter shape mismatch for '{source_spec.name}': "
                f"source={source_spec.shape}, "
                f"target={target_spec.shape}."
            )

        if source_spec.dtype != target_spec.dtype:
            raise ParameterContractError(
                f"Parameter dtype mismatch for '{source_spec.name}': "
                f"source={source_spec.dtype}, "
                f"target={target_spec.dtype}."
            )