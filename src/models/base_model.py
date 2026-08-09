"""
Abstract model interface for FedMed.

This module defines BaseModel, the contract that every model
implementation in FedMed must satisfy.

The purpose of this abstraction is to ensure that the training,
evaluation, and federated-learning layers do not depend directly
on a specific model architecture.

Concrete implementations may represent:

- Basic PyTorch models used during development
- MONAI-based medical models
- Future model architectures

This module intentionally does NOT contain:

- Training loops
- Loss functions
- Optimizers
- Dataset or DataLoader logic
- Flower-specific implementation
- Hardcoded medical architectures
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import nn

from src.common.exceptions import ModelError


class BaseModel(ABC):
    """
    Abstract base class for all FedMed model implementations.

    Subclasses are responsible for constructing the underlying
    ``torch.nn.Module`` through ``build()``.

    The base class provides common functionality for:

    - Model construction
    - Forward passes
    - Model state extraction
    - Model state loading
    - Device management
    - Training/evaluation mode
    - Checkpoint compatibility
    - Model metadata
    """

    def __init__(
        self,
        name: str,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the model.

        Args:
            name:
                Human-readable model identifier.

            device:
                PyTorch device string such as ``cpu``, ``cuda``,
                or ``cuda:0``.

        Raises:
            ModelError:
                If the model cannot be constructed.
        """

        self.name = name
        self._device = torch.device(device)

        try:
            self._network = self.build()
        except Exception as exc:
            raise ModelError(
                f"Failed to build model '{self.name}': {exc}"
            ) from exc

        if not isinstance(self._network, nn.Module):
            raise ModelError(
                f"Model '{self.name}' build() must return "
                f"a torch.nn.Module."
            )

        self._network.to(self._device)

    # ------------------------------------------------------------------
    # Subclass responsibility
    # ------------------------------------------------------------------

    @abstractmethod
    def build(self) -> nn.Module:
        """
        Construct the underlying PyTorch network.

        Subclasses must implement this method.

        Returns:
            An instantiated ``torch.nn.Module``.
        """

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute a forward pass.

        Args:
            x:
                Input tensor.

        Returns:
            Model output tensor.
        """

        return self._network(x)

    # ------------------------------------------------------------------
    # Model state exchange
    # ------------------------------------------------------------------

    def get_parameters(self) -> list[np.ndarray]:
        """
        Extract the complete model state as NumPy arrays.

        The state includes both trainable parameters and persistent
        buffers contained in the model's state_dict.

        NumPy arrays provide a framework-neutral representation that
        can later be consumed by the federated-learning layer.

        Returns:
            Model state tensors represented as NumPy arrays, in
            state_dict order.
        """

        return [
            tensor.detach().cpu().numpy().copy()
            for tensor in self._network.state_dict().values()
        ]

    def set_parameters(
        self,
        parameters: list[np.ndarray],
    ) -> None:
        """
        Load model state from NumPy arrays.

        Args:
            parameters:
                NumPy arrays matching the model state_dict order,
                shapes, and number of tensors.

        Raises:
            ModelError:
                If the supplied state does not match the model.
        """

        current_state = self._network.state_dict()
        current_keys = list(current_state.keys())

        if len(parameters) != len(current_keys):
            raise ModelError(
                f"Model '{self.name}' state count mismatch: "
                f"expected {len(current_keys)}, "
                f"received {len(parameters)}."
            )

        new_state: dict[str, torch.Tensor] = {}

        for key, value in zip(current_keys, parameters):
            expected_tensor = current_state[key]

            if tuple(value.shape) != tuple(expected_tensor.shape):
                raise ModelError(
                    f"Shape mismatch for '{key}' in model "
                    f"'{self.name}': "
                    f"expected {tuple(expected_tensor.shape)}, "
                    f"received {tuple(value.shape)}."
                )

            tensor = torch.from_numpy(value).to(
                device=self._device,
                dtype=expected_tensor.dtype,
            )

            new_state[key] = tensor

        try:
            self._network.load_state_dict(
                new_state,
                strict=True,
            )
        except RuntimeError as exc:
            raise ModelError(
                f"Failed to load state for model '{self.name}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Checkpoint compatibility
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Return the model state dictionary.

        This is intended for checkpointing and local PyTorch
        persistence.
        """

        return self._network.state_dict()

    def load_state_dict(
        self,
        state: dict[str, torch.Tensor],
    ) -> None:
        """
        Load a PyTorch state dictionary.

        Args:
            state:
                State dictionary previously produced by ``state_dict``.
        """

        try:
            self._network.load_state_dict(
                state,
                strict=True,
            )
        except RuntimeError as exc:
            raise ModelError(
                f"Failed to load checkpoint for model "
                f"'{self.name}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Training / evaluation mode
    # ------------------------------------------------------------------

    def train_mode(self) -> None:
        """Set the model to training mode."""

        self._network.train()

    def eval_mode(self) -> None:
        """Set the model to evaluation mode."""

        self._network.eval()

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device: str) -> None:
        """
        Move the model to another device.

        Args:
            device:
                Target PyTorch device.
        """

        try:
            self._device = torch.device(device)
            self._network.to(self._device)
        except (RuntimeError, ValueError) as exc:
            raise ModelError(
                f"Failed to move model '{self.name}' "
                f"to device '{device}': {exc}"
            ) from exc

    @property
    def device(self) -> torch.device:
        """Return the current model device."""

        return self._device

    # ------------------------------------------------------------------
    # Underlying network
    # ------------------------------------------------------------------

    @property
    def network(self) -> nn.Module:
        """
        Return the underlying PyTorch network.

        The training layer may use this to construct optimizers
        and perform framework-specific operations.

        Other layers should prefer the BaseModel interface.
        """

        return self._network

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        """
        Return the number of trainable parameters.
        """

        return sum(
            parameter.numel()
            for parameter in self._network.parameters()
            if parameter.requires_grad
        )

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Return basic model metadata.

        This will later support experiment tracking,
        monitoring, and federated-learning diagnostics.
        """

        return {
            "name": self.name,
            "device": str(self._device),
            "num_parameters": self.num_parameters,
        }