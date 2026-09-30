"""
model/simple_model.py
=====================
FedMed demonstration models for federated learning.

Two models are provided:

1. **MedMLP** — A multi-layer perceptron for tabular data (e.g. patient
   risk-score prediction from structured EHR features).

2. **MedCNN** — A lightweight convolutional network for medical imaging
   (e.g. binary chest-X-ray classification from 64×64 grey-scale inputs).

Both expose the same interface so the FL client can swap them by config:

    model = MedMLP(input_dim=20, hidden_dim=64, output_dim=2)
    # or
    model = MedCNN(in_channels=1, num_classes=2)

    params = get_parameters(model)   # → list[np.ndarray]
    set_parameters(model, params)    # load weights back
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Tabular model — patient risk-score MLP
# ---------------------------------------------------------------------------


class MedMLP(nn.Module):
    """Feed-forward MLP for tabular patient data.

    Architecture: Linear → BN → ReLU → Dropout → Linear → BN → ReLU
                  → Dropout → Linear (output)

    Parameters
    ----------
    input_dim:
        Number of input features (EHR columns, lab values, etc.).
    hidden_dim:
        Width of the two hidden layers.
    output_dim:
        Number of output classes (2 for binary, >2 for multi-class).
    dropout:
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 64,
        output_dim: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. Imaging model — lightweight CNN for chest X-ray
# ---------------------------------------------------------------------------


class MedCNN(nn.Module):
    """Lightweight CNN for binary medical image classification.

    Accepts grey-scale images of shape ``(B, C, H, W)`` where H = W = 64
    by default. Adjust ``in_channels`` for multi-channel inputs.

    Architecture: Conv → BN → ReLU → MaxPool (×3) → Flatten → FC → ReLU
                  → FC (output)

    Parameters
    ----------
    in_channels:
        Number of input channels (1 for grey-scale, 3 for RGB).
    num_classes:
        Number of output classes.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (B, C, 64, 64) → (B, 32, 32, 32)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: (B, 32, 32, 32) → (B, 64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: (B, 64, 16, 16) → (B, 128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Parameter helpers — shared by FL client and server
# ---------------------------------------------------------------------------


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract all trainable parameters as a list of NumPy arrays.

    This is the format expected by Flower's ``NDArrays`` type and by the
    FedMed SecAgg masking layer.
    """
    return [val.cpu().detach().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load a list of NumPy arrays back into *model* in-place.

    Parameters must be in the same order as returned by :func:`get_parameters`.
    """
    state_dict = model.state_dict()
    new_state = {
        k: torch.tensor(v)
        for k, v in zip(state_dict.keys(), parameters)
    }
    model.load_state_dict(new_state, strict=True)


def flatten_parameters(parameters: List[np.ndarray]) -> np.ndarray:
    """Concatenate all parameter arrays into one 1-D float64 vector.

    Used by the SecAgg and defense layers, which operate on flat vectors.
    """
    return np.concatenate([p.flatten().astype(np.float64) for p in parameters])


def unflatten_parameters(
    flat: np.ndarray, shapes: List[tuple]
) -> List[np.ndarray]:
    """Reconstruct a list of parameter arrays from a flat vector.

    Parameters
    ----------
    flat:
        1-D array produced by :func:`flatten_parameters`.
    shapes:
        List of tuples specifying the original shape of each parameter tensor.
    """
    arrays = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        arrays.append(flat[offset : offset + size].reshape(shape).astype(np.float32))
        offset += size
    return arrays
