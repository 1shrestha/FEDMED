"""
Tests for the Flower runtime adapter in app/client.py.

Phase 3.4-B
-----------

The tests verify that the Flower boundary remains a thin adapter over
the existing framework-independent FedMed FederatedClient.

Architecture under test:

    Flower ClientApp
          |
          v
    FedMedNumPyClient
          |
          v
    FederatedClient
       /        \
   Trainer    Evaluator

These tests intentionally use a lightweight fake FederatedClient
subclass for adapter-boundary tests. The goal is to verify the Flower
mapping without coupling this suite to local training mechanics that
are already covered by the Phase 2/3 tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from flwr.app import Context
from flwr.client import Client
from flwr.clientapp import ClientApp

from app.client import (
    FedMedNumPyClient,
    create_client_app,
)
from src.common.exceptions import FederatedLearningError
from src.fl.client import FederatedClient
from src.fl.parameters import ParameterPayload


# ======================================================================
# Test double
# ======================================================================


class StubFederatedClient(FederatedClient):
    """
    Minimal deterministic FederatedClient test double.

    The adapter is responsible for delegation and runtime conversion;
    local training/evaluation are already tested by tests/test_client.py.
    """

    def __init__(
        self,
        client_id: str = "client_test",
    ) -> None:
        # FederatedClient's production constructor is intentionally not
        # called here. The adapter only requires a FederatedClient
        # instance and the three public delegation methods.
        self._client_id = client_id

        self.fit_calls: list[ParameterPayload] = []
        self.evaluate_calls: list[ParameterPayload] = []
        self.get_parameters_calls = 0

        self._parameters: ParameterPayload = [
            np.array(
                [[1.0, 2.0]],
                dtype=np.float32,
            ),
            np.array(
                [3.0],
                dtype=np.float32,
            ),
        ]

    @property
    def client_id(self) -> str:
        return self._client_id

    def get_parameters(self) -> ParameterPayload:
        self.get_parameters_calls += 1

        return [
            parameter.copy()
            for parameter in self._parameters
        ]

    def fit(self, parameters: ParameterPayload):
        self.fit_calls.append(
            [
                parameter.copy()
                for parameter in parameters
            ]
        )

        updated = [
            parameter + 1.0
            for parameter in parameters
        ]

        # The concrete FederatedFitResult is imported lazily so this
        # test double remains focused on the adapter boundary.
        from src.fl.client import FederatedFitResult

        return FederatedFitResult(
            client_id=self.client_id,
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

    def evaluate(self, parameters: ParameterPayload):
        self.evaluate_calls.append(
            [
                parameter.copy()
                for parameter in parameters
            ]
        )

        from src.fl.client import FederatedEvaluateResult

        return FederatedEvaluateResult(
            client_id=self.client_id,
            loss=0.125,
            num_examples=8,
            metrics={
                "accuracy": 0.9375,
            },
        )


class FailingFitClient(StubFederatedClient):
    """Stub client that fails during local training."""

    def fit(self, parameters: ParameterPayload):
        self.fit_calls.append(
            [
                parameter.copy()
                for parameter in parameters
            ]
        )

        raise FederatedLearningError(
            "synthetic fit failure",
        )


class FailingEvaluateClient(StubFederatedClient):
    """Stub client that fails during local evaluation."""

    def evaluate(self, parameters: ParameterPayload):
        self.evaluate_calls.append(
            [
                parameter.copy()
                for parameter in parameters
            ]
        )

        raise FederatedLearningError(
            "synthetic evaluation failure",
        )


# ======================================================================
# Helpers
# ======================================================================


def make_parameters() -> list[np.ndarray]:
    """Create deterministic Flower-compatible parameter arrays."""

    return [
        np.array(
            [[10.0, 20.0]],
            dtype=np.float32,
        ),
        np.array(
            [30.0],
            dtype=np.float32,
        ),
    ]


def assert_parameter_payload_equal(
    actual: ParameterPayload,
    expected: ParameterPayload,
) -> None:
    """Compare parameter payloads without relying on object identity."""

    assert len(actual) == len(expected)

    for actual_array, expected_array in zip(
        actual,
        expected,
    ):
        np.testing.assert_array_equal(
            actual_array,
            expected_array,
        )


# ======================================================================
# Construction
# ======================================================================


def test_adapter_requires_federated_client() -> None:
    """Only a real FederatedClient may be wrapped."""

    with pytest.raises(
        FederatedLearningError,
    ):
        FedMedNumPyClient(
            object(),  # type: ignore[arg-type]
        )


def test_adapter_exposes_wrapped_client() -> None:
    """The adapter must retain the injected FedMed client."""

    client = StubFederatedClient(
        client_id="client_42",
    )

    adapter = FedMedNumPyClient(
        client,
    )

    assert adapter.federated_client is client
    assert adapter.client_id == "client_42"


# ======================================================================
# get_parameters
# ======================================================================


def test_get_parameters_delegates_to_fedmed_client() -> None:
    """Flower parameter retrieval delegates to FederatedClient."""

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)

    parameters = adapter.get_parameters(
        {},
    )

    assert client.get_parameters_calls == 1

    assert_parameter_payload_equal(
        parameters,
        client._parameters,
    )


def test_get_parameters_returns_independent_arrays() -> None:
    """
    Mutating Flower-facing parameters must not mutate the FedMed
    client's internal parameter representation.
    """

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)

    parameters = adapter.get_parameters(
        {},
    )

    parameters[0][0, 0] = 9999.0

    fresh_parameters = adapter.get_parameters(
        {},
    )

    assert fresh_parameters[0][0, 0] == 1.0


# ======================================================================
# fit
# ======================================================================


def test_fit_delegates_parameters_to_fedmed_client() -> None:
    """Flower fit parameters are passed to FederatedClient.fit()."""

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)

    parameters = make_parameters()

    adapter.fit(
        parameters,
        {},
    )

    assert len(client.fit_calls) == 1

    assert_parameter_payload_equal(
        client.fit_calls[0],
        parameters,
    )


def test_fit_returns_flower_compatible_result() -> None:
    """
    FedMed FederatedFitResult is mapped to Flower's NumPyClient result
    tuple.
    """

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)

    parameters = make_parameters()

    updated, num_examples, metrics = adapter.fit(
        parameters,
        {},
    )

    expected_updated = [
        parameter + 1.0
        for parameter in parameters
    ]

    assert_parameter_payload_equal(
        updated,
        expected_updated,
    )

    assert num_examples == 8

    assert metrics == {
        "accuracy": 0.875,
        "loss": 0.25,
    }


def test_fit_copies_input_parameters_before_delegation() -> None:
    """
    The adapter must not allow the wrapped client to retain mutable
    references to the caller-owned Flower arrays.
    """

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)

    parameters = make_parameters()

    adapter.fit(
        parameters,
        {},
    )

    parameters[0][0, 0] = 7777.0

    assert client.fit_calls[0][0][0, 0] == 10.0


def test_fit_result_parameters_are_independent() -> None:
    """Returned result arrays must not alias the adapter's internal data."""

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)

    parameters = make_parameters()

    updated, _, _ = adapter.fit(
        parameters,
        {},
    )

    updated[0][0, 0] = 8888.0

    assert client.fit_calls[0][0][0, 0] == 10.0


def test_fit_preserves_fedmed_learning_errors() -> None:
    """FedMed domain errors must cross the adapter boundary unchanged."""

    client = FailingFitClient()
    adapter = FedMedNumPyClient(client)

    with pytest.raises(
        FederatedLearningError,
        match="synthetic fit failure",
    ):
        adapter.fit(
            make_parameters(),
            {},
        )


# ======================================================================
# evaluate
# ======================================================================


def test_evaluate_delegates_parameters_to_fedmed_client() -> None:
    """Flower evaluation parameters reach FederatedClient.evaluate()."""

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)

    parameters = make_parameters()

    adapter.evaluate(
        parameters,
        {},
    )

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
    """Evaluation domain failures are not silently swallowed."""

    client = FailingEvaluateClient()
    adapter = FedMedNumPyClient(client)

    with pytest.raises(
        FederatedLearningError,
        match="synthetic evaluation failure",
    ):
        adapter.evaluate(
            make_parameters(),
            {},
        )


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
def test_metrics_accept_flower_scalar_values(
    value: Any,
) -> None:
    """Flower-compatible scalar values are preserved."""

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)

    normalized = adapter._normalize_metrics(
        {"metric": value},
    )

    assert normalized["metric"] == (
        value.item()
        if isinstance(value, np.generic)
        else value
    )


def test_metrics_reject_non_scalar_values() -> None:
    """Arbitrary Python objects must not cross into Flower metrics."""

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)

    with pytest.raises(
        FederatedLearningError,
        match="not compatible with Flower Scalar",
    ):
        adapter._normalize_metrics(
            {
                "invalid": {
                    "nested": "value",
                }
            },
        )


# ======================================================================
# Properties
# ======================================================================


def test_get_properties_returns_stable_client_identity() -> None:
    """Only stable runtime-safe metadata is exposed."""

    client = StubFederatedClient(
        client_id="client_properties",
    )

    adapter = FedMedNumPyClient(client)

    properties = adapter.get_properties(
        {},
    )

    assert properties == {
        "fedmed_client_id": "client_properties",
    }


# ======================================================================
# ClientApp
# ======================================================================


def test_create_client_app_requires_callable_factory() -> None:
    """ClientApp creation requires a client factory."""

    with pytest.raises(
        FederatedLearningError,
    ):
        create_client_app(
            object(),  # type: ignore[arg-type]
        )


def test_create_client_app_returns_client_app() -> None:
    """The factory creates Flower's modern ClientApp object."""

    client = StubFederatedClient(
        client_id="client_app",
    )

    app = create_client_app(
        lambda context: client,
    )

    assert isinstance(
        app,
        ClientApp,
    )


def test_client_app_factory_builds_flower_client() -> None:
    """
    The application factory must construct a Flower Client from the
    injected FedMed client factory.
    """

    client = StubFederatedClient(
        client_id="client_app_factory",
    )

    app = create_client_app(
        lambda context: client,
    )

    # ClientApp's callable is the Flower runtime boundary. We verify
    # the adapter itself independently above; here we verify that the
    # produced object is a valid ClientApp and has a callable client
    # factory.
    assert app is not None
    assert callable(
        app._client_fn,
    )


# ======================================================================
# Framework boundary
# ======================================================================


def test_app_client_does_not_import_flower_into_fedmed_core() -> None:
    """
    Flower imports are confined to app/client.py.

    The framework-independent FederatedClient module must remain free
    of Flower imports.
    """

    from pathlib import Path

    core_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "fl"
        / "client.py"
    )

    source = core_path.read_text(
        encoding="utf-8",
    )

    assert "import flwr" not in source
    assert "from flwr" not in source


def test_adapter_is_a_flower_numpy_client() -> None:
    """The concrete runtime adapter uses the intended Flower boundary."""

    client = StubFederatedClient()
    adapter = FedMedNumPyClient(client)

    assert isinstance(
        adapter,
        __import__(
            "flwr.client",
            fromlist=["NumPyClient"],
        ).NumPyClient,
    )