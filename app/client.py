from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from flwr.app import Context
from flwr.client import NumPyClient
from flwr.clientapp import ClientApp

from src.common.exceptions import FederatedLearningError
from src.fl.client import FederatedClient
from src.fl.parameters import ParameterPayload


FlowerScalar = bool | bytes | float | int | str
FlowerMetrics = dict[str, FlowerScalar]


class FedMedNumPyClient(NumPyClient):
    """Thin Flower adapter around an existing FedMed FederatedClient."""

    def __init__(self, federated_client: FederatedClient) -> None:
        if not isinstance(federated_client, FederatedClient):
            raise FederatedLearningError(
                "FedMedNumPyClient requires a FederatedClient instance, "
                f"got {type(federated_client).__name__}."
            )
        self._federated_client = federated_client

    @property
    def federated_client(self) -> FederatedClient:
        """Return the wrapped framework-independent FedMed client."""
        return self._federated_client

    @property
    def client_id(self) -> str:
        """Return the stable FedMed client identifier."""
        return self._federated_client.client_id

    def get_parameters(
        self,
        config: Mapping[str, FlowerScalar],
    ) -> list[np.ndarray]:
        """Delegate parameter extraction to FederatedClient."""
        del config
        return self._federated_client.get_parameters()

    def fit(
        self,
        parameters: list[np.ndarray],
        config: Mapping[str, FlowerScalar],
    ) -> tuple[list[np.ndarray], int, FlowerMetrics]:
        """Delegate local training to FederatedClient."""
        del config

        result = self._federated_client.fit(
            self._copy_parameters(parameters)
        )

        return (
            self._copy_parameters(result.parameters),
            int(result.num_examples),
            self._normalize_metrics(result.metrics),
        )

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: Mapping[str, FlowerScalar],
    ) -> tuple[float, int, FlowerMetrics]:
        """Delegate local evaluation to FederatedClient."""
        del config

        result = self._federated_client.evaluate(
            self._copy_parameters(parameters)
        )

        return (
            float(result.loss),
            int(result.num_examples),
            self._normalize_metrics(result.metrics),
        )

    def get_properties(
        self,
        config: Mapping[str, FlowerScalar],
    ) -> FlowerMetrics:
        """Expose only stable runtime-safe FedMed metadata."""
        del config
        return {"fedmed_client_id": self.client_id}

    @staticmethod
    def _copy_parameters(
        parameters: list[np.ndarray],
    ) -> ParameterPayload:
        """Copy arrays at the Flower/FedMed boundary."""
        return [
            np.array(parameter, copy=True)
            for parameter in parameters
        ]

    @staticmethod
    def _normalize_metrics(
        metrics: Mapping[str, Any],
    ) -> FlowerMetrics:
        """Convert FedMed metrics to Flower-compatible scalar values."""
        normalized: FlowerMetrics = {}

        for name, value in metrics.items():
            key = str(name)

            if isinstance(value, (bool, bytes, float, int, str)):
                normalized[key] = value
                continue

            if isinstance(value, np.generic):
                scalar = value.item()
                if isinstance(
                    scalar,
                    (bool, bytes, float, int, str),
                ):
                    normalized[key] = scalar
                    continue

            raise FederatedLearningError(
                "FedMed metric is not compatible with Flower Scalar: "
                f"'{key}' has type {type(value).__name__}."
            )

        return normalized


FederatedClientFactory = Callable[[Context], FederatedClient]


def create_client_app(
    client_factory: FederatedClientFactory,
) -> ClientApp:
    """Create a Flower ClientApp around a FedMed client factory.

    Construction of the model, datasets, Trainer, Evaluator and
    FederatedClient remains outside this adapter. That assembly belongs
    to the later configuration/application-assembly phase.
    """
    if not callable(client_factory):
        raise FederatedLearningError(
            "client_factory must be callable, "
            f"got {type(client_factory).__name__}."
        )

    def client_fn(context: Context):
        federated_client = client_factory(context)
        return FedMedNumPyClient(
            federated_client
        ).to_client()

    return ClientApp(client_fn=client_fn)


__all__ = [
    "FedMedNumPyClient",
    "FederatedClientFactory",
    "create_client_app",
]