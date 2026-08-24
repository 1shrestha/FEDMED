"""
Tests for the Flower runtime adapter in app/client.py.

The tests keep Flower-specific behavior at the application boundary and
verify that the existing framework-independent FedMed contracts are
delegated to correctly.

These tests intentionally do not inspect private Flower ClientApp
attributes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from flwr.client import NumPyClient
from flwr.clientapp import ClientApp

from app.client import FedMedNumPyClient, create_client_app
from src.common.exceptions import FederatedLearningError
from src.fl.client import (
    FederatedClient,
    FederatedEvaluateResult,
    FederatedFitResult,
)
from src.fl.parameters import ParameterPayload


# ======================================================================
# Test doubles
# ======================================================================


class StubFederatedClient(FederatedClient):
    """Minimal deterministic test double for the Flower adapter."""

    def __init__(self, client_id: str = "client_test") -> None:
        # Deliberately avoid FederatedClient.__init__.
        # Adapter tests only require the public delegation surface.
        self._client_id = client_id

        self.fit_calls: list[ParameterPayload] = []
        self.evaluate_calls: list[ParameterPayload] = []
        self.get_parameters_calls = 0

        self._parameters: ParameterPayload = [
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([3.0], dtype=np.float32),
        ]

    @property
    def client_id(self) -> str:
        return self._client_id

    def get_parameters(self) -> ParameterPayload:
        self.get_parameters_calls += 1
        return [parameter.copy() for parameter in self._parameters]

    def fit(self, parameters: ParameterPayload) -> FederatedFitResult:
        self.fit_calls.append(
            [parameter.copy() for parameter in parameters]
        )

        updated = [parameter + 1.0 for parameter in parameters]

        # FederatedFitResult is intentionally constructed according to
        # the existing FedMed contract. It has no client_id field.
        return FederatedFitResult(
            parameters=updated,
            num_examples=8,
            metrics={
                "accuracy": 0.875,
                "loss": 0.25,
            },
            epochs_completed=1,
            batches_processed=2,
            final_loss=0.25,
        )

    def evaluate(
        self,
        parameters: ParameterPayload,
    ) -> FederatedEvaluateResult:
        self.evaluate_calls.append(
            [parameter.copy() for parameter in parameters]
        )

        # FederatedEvaluateResult is intentionally constructed according
        # to the existing FedMed contract. It has no client_id field.
        return FederatedEvaluateResult(
            loss=0.125,
            num_examples=8,
            metrics={
                "accuracy": 0.9375,
            },
        )


class FailingFitClient(StubFederatedClient):
    """Test double that fails during local training."""

    def fit(self, parameters: ParameterPayload) -> FederatedFitResult:
        self.fit_calls.append(
            [parameter.copy() for parameter in parameters]
        )
        raise FederatedLearningError("synthetic fit failure")


class FailingEvaluateClient(StubFederatedClient):
    """Test double that fails during local evaluation."""

    def evaluate(
        self,
        parameters: ParameterPayload,
    ) -> FederatedEvaluateResult:
        self.evaluate_calls.append(
            [parameter.copy() for parameter in parameters]
        )
        raise FederatedLearningError("synthetic evaluation failure")


# ======================================================================
# Helpers
# ======================================================================


def make_parameters() -> ParameterPayload:
    """Create deterministic Flower-compatible parameter arrays."""

    return [
        np.array([[10.0, 20.0]], dtype=np.float32),
        np.array([30.0], dtype=np.float32),
    ]


def assert_parameter_payload_equal(
    actual: ParameterPayload,
    expected: ParameterPayload,
) -> None:
    """Compare parameter arrays without relying on object identity."""

    assert len(actual) == len(expected)

    for actual_array, expected_array in zip(actual, expected):
        np.testing.assert_array_equal(
            actual_array,
            expected_array,
        )


# ======================================================================
# Construction
# ======================================================================


def test_adapter_requires_federated_client() -> None:
    """Only a FederatedClient may be wrapped."""

    with pytest.raises(FederatedLearningError):
        FedMedNumPyClient(
            object(),  # type: ignore[arg-type]
        )


def test_adapter_exposes_wrapped_client() -> None:
    """The adapter retains the injected FedMed client."""

    client = StubFederatedClient("client_42")
    adapter = FedMedNumPyClient(client)

    assert adapter.federated_client is client
    assert adapter.client_id == "client_42"


def test_adapter_is_flower_numpy_client() -> None:
    """The runtime boundary is Flower's NumPyClient."""

    adapter = FedMedNumPyClient(StubFederatedClient())

    assert isinstance(adapter, NumPyClient)


# ======================================================================
# get_parameters
# ======================================================================


def test_get_parameters_delegates_to_fedmed_client() -> None:
    """Flower parameter retrieval delegates to FedMed."""

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)

    parameters = adapter.get_parameters({})

    assert client.get_parameters_calls == 1
    assert_parameter_payload_equal(
        parameters,
        client._parameters,
    )


def test_get_parameters_returns_independent_arrays() -> None:
    """Flower callers cannot mutate FedMed's stored parameters."""

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)

    parameters = adapter.get_parameters({})
    parameters[0][0, 0] = 9999.0

    fresh_parameters = adapter.get_parameters({})

    assert fresh_parameters[0][0, 0] == 1.0


# ======================================================================
# fit
# ======================================================================


def test_fit_delegates_parameters_to_fedmed_client() -> None:
    """Flower fit parameters reach FederatedClient.fit()."""

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)
    parameters = make_parameters()

    adapter.fit(parameters, {})

    assert len(client.fit_calls) == 1
    assert_parameter_payload_equal(
        client.fit_calls[0],
        parameters,
    )


def test_fit_returns_flower_compatible_result() -> None:
    """FedMed fit results map to Flower's NumPyClient tuple."""

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)

    updated, num_examples, metrics = adapter.fit(
        make_parameters(),
        {},
    )

    expected = [
        np.array([[11.0, 21.0]], dtype=np.float32),
        np.array([31.0], dtype=np.float32),
    ]

    assert_parameter_payload_equal(updated, expected)
    assert num_examples == 8
    assert metrics == {
        "accuracy": 0.875,
        "loss": 0.25,
    }


def test_fit_copies_input_parameters_before_delegation() -> None:
    """The wrapped client does not retain caller-owned arrays."""

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)
    parameters = make_parameters()

    adapter.fit(parameters, {})

    parameters[0][0, 0] = 7777.0

    assert client.fit_calls[0][0][0, 0] == 10.0


def test_fit_result_parameters_are_independent() -> None:
    """Returned fit parameters do not alias the input arrays."""

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)
    parameters = make_parameters()

    updated, _, _ = adapter.fit(parameters, {})

    updated[0][0, 0] = 8888.0

    assert parameters[0][0, 0] == 10.0


def test_fit_preserves_fedmed_learning_errors() -> None:
    """FedMed domain errors cross the adapter boundary unchanged."""

    adapter = FedMedNumPyClient(FailingFitClient())

    with pytest.raises(
        FederatedLearningError,
        match="synthetic fit failure",
    ):
        adapter.fit(make_parameters(), {})


# ======================================================================
# evaluate
# ======================================================================


def test_evaluate_delegates_parameters_to_fedmed_client() -> None:
    """Flower evaluation parameters reach FederatedClient.evaluate()."""

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)
    parameters = make_parameters()

    adapter.evaluate(parameters, {})

    assert len(client.evaluate_calls) == 1
    assert_parameter_payload_equal(
        client.evaluate_calls[0],
        parameters,
    )


def test_evaluate_returns_flower_compatible_result() -> None:
    """FedMed evaluation results map to Flower's expected tuple."""

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)

    loss, num_examples, metrics = adapter.evaluate(
        make_parameters(),
        {},
    )

    assert loss == pytest.approx(0.125)
    assert num_examples == 8
    assert metrics == {
        "accuracy": 0.9375,
    }


def test_evaluate_preserves_fedmed_learning_errors() -> None:
    """Evaluation failures are not silently swallowed."""

    adapter = FedMedNumPyClient(FailingEvaluateClient())

    with pytest.raises(
        FederatedLearningError,
        match="synthetic evaluation failure",
    ):
        adapter.evaluate(make_parameters(), {})


# ======================================================================
# Metrics boundary
# ======================================================================


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        1,
        1.5,
        "text",
        b"bytes",
        np.int64(4),
        np.float32(0.5),
    ],
)
def test_metrics_accept_flower_scalar_values(value: Any) -> None:
    """Flower-compatible scalar values are accepted."""

    adapter = FedMedNumPyClient(StubFederatedClient())

    normalized = adapter._normalize_metrics({"metric": value})

    expected = value.item() if isinstance(value, np.generic) else value

    assert normalized["metric"] == expected


def test_metrics_reject_non_scalar_values() -> None:
    """Arbitrary nested values cannot cross into Flower metrics."""

    adapter = FedMedNumPyClient(StubFederatedClient())

    with pytest.raises(
        FederatedLearningError,
        match="not compatible with Flower Scalar",
    ):
        adapter._normalize_metrics(
            {
                "invalid": {
                    "nested": "value",
                }
            }
        )


# ======================================================================
# Properties
# ======================================================================


def test_get_properties_returns_stable_client_identity() -> None:
    """The adapter exposes stable FedMed client identity metadata."""

    client = StubFederatedClient("client_properties")
    adapter = FedMedNumPyClient(client)

    properties = adapter.get_properties({})

    assert properties == {
        "fedmed_client_id": "client_properties",
    }


# ======================================================================
# ClientApp
# ======================================================================


def test_create_client_app_requires_callable_factory() -> None:
    """ClientApp creation requires a callable factory."""

    with pytest.raises(FederatedLearningError):
        create_client_app(
            object(),  # type: ignore[arg-type]
        )


def test_create_client_app_returns_client_app() -> None:
    """The application factory creates Flower's ClientApp."""

    client = StubFederatedClient("client_app")

    app = create_client_app(
        lambda context: client,
    )

    assert isinstance(app, ClientApp)


def test_client_app_can_be_created_from_fedmed_factory() -> None:
    """
    A valid FedMed client factory can be accepted by create_client_app.

    This intentionally does not inspect private ClientApp internals.
    Flower owns the runtime invocation mechanism.
    """

    created: list[str] = []

    def factory(context: Any) -> FederatedClient:
        created.append("called")
        return StubFederatedClient("client_factory")

    app = create_client_app(factory)

    assert isinstance(app, ClientApp)
    assert created == []


# ======================================================================
# Framework boundary
# ======================================================================


def test_fedmed_core_client_has_no_flower_dependency() -> None:
    """
    The framework-independent FedMed client must not import Flower.

    Flower remains an application/runtime integration concern.
    """

    core_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "fl"
        / "client.py"
    )

    source = core_path.read_text(encoding="utf-8")

    assert "import flwr" not in source
    assert "from flwr" not in source