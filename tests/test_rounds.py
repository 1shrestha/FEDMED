"""
Tests for the FedMed federated round lifecycle and coordinator.

Phase 3.3-A coverage:

- RoundState lifecycle contract
- ClientFailure validation
- RoundResult validation and immutability
- RoundExecution defensive parameter handling
- RoundCoordinator client validation
- successful federated training round
- client-level training failure
- optional evaluation
- unknown client selection
- invalid parameter payloads
- invalid round numbers
- invalid strategy results
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.common.exceptions import FederatedLearningError
from src.fl.client import (
    FederatedClient,
    FederatedEvaluateResult,
    FederatedFitResult,
)
from src.fl.parameters import ParameterContract
from src.fl.rounds import (
    ClientFailure,
    RoundCoordinator,
    RoundExecution,
    RoundResult,
    RoundState,
)


# ============================================================
# Test doubles
# ============================================================


class FakeModel:
    """
    Minimal model-like object used only where a real model is
    unnecessary for the round contract tests.
    """

    name = "fake-model"


class FakeStrategy:
    """
    Minimal strategy implementation for Phase 3.3-A tests.

    The public strategy contract will be finalized in Phase 3.3-B.
    This test double implements only the structural operations
    currently required by RoundCoordinator.
    """

    def __init__(
        self,
        fit_clients: list[FederatedClient] | None = None,
        evaluate_clients: list[FederatedClient] | None = None,
        aggregated_parameters: list[np.ndarray] | None = None,
    ) -> None:
        self.fit_clients = fit_clients or []
        self.evaluate_clients = evaluate_clients or []
        self.aggregated_parameters = (
            aggregated_parameters
            if aggregated_parameters is not None
            else [np.array([2.0], dtype=np.float32)]
        )

        self.fit_calls: list[int] = []
        self.aggregate_calls: list[int] = []
        self.evaluate_selection_calls: list[int] = []

    def select_fit_clients(
        self,
        clients,
        round_number: int,
    ):
        self.fit_calls.append(round_number)

        if self.fit_clients:
            return self.fit_clients

        return list(clients)

    def aggregate_fit(
        self,
        results,
        round_number: int,
    ):
        self.aggregate_calls.append(round_number)

        assert results

        return [
            parameter.copy()
            for parameter in self.aggregated_parameters
        ]

    def select_evaluate_clients(
        self,
        clients,
        round_number: int,
    ):
        self.evaluate_selection_calls.append(round_number)

        if self.evaluate_clients:
            return self.evaluate_clients

        return list(clients)


class FailingFitClient(FederatedClient):
    """Federated client whose training operation fails."""

    def fit(self, parameters):
        raise FederatedLearningError(
            f"Training failed for client '{self.client_id}'."
        )


class FailingEvaluateClient(FederatedClient):
    """Federated client whose evaluation operation fails."""

    def evaluate(self, parameters):
        raise FederatedLearningError(
            f"Evaluation failed for client '{self.client_id}'."
        )


# ============================================================
# Helpers
# ============================================================


def make_client(
    client_id: str,
    *,
    model=None,
    evaluator=None,
) -> FederatedClient:
    """
    Create a real FederatedClient for coordinator tests.

    The helper intentionally uses the actual Phase 3.2 client
    abstraction rather than replacing it with a mock object.
    """

    pytest.importorskip("torch")

    import torch
    import torch.nn as nn

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(1, 1)

    class TestBaseModel:
        name = "test-model"

        def __init__(self):
            self.model = TestModel()

        def state_dict(self):
            return self.model.state_dict()

        def get_parameters(self):
            return [
                value.detach().cpu().numpy().copy()
                for value in self.model.state_dict().values()
            ]

        def set_parameters(self, parameters):
            state = self.model.state_dict()

            for (name, tensor), parameter in zip(
                state.items(),
                parameters,
            ):
                tensor.copy_(
                    torch.from_numpy(parameter)
                )

    base_model = TestBaseModel()

    # We use the real ParameterContract machinery.
    contract = ParameterContract.from_model(base_model)

    # Constructing FederatedClient depends on the exact Phase 3.2
    # constructor, so this helper is intentionally isolated.
    return FederatedClient(
        client_id=client_id,
        model=base_model,
        trainer=None,
        evaluator=evaluator,
        parameter_contract=contract,
    )


def make_parameters(
    value: float = 1.0,
) -> list[np.ndarray]:
    """Create the test parameter payload."""

    return [
        np.array(
            [[value]],
            dtype=np.float32,
        ),
        np.array(
            [value],
            dtype=np.float32,
        ),
    ]


# ============================================================
# RoundState tests
# ============================================================


class TestRoundState:
    """Tests for the round lifecycle enumeration."""

    def test_all_expected_states_exist(self):
        assert RoundState.CREATED.value == "created"
        assert RoundState.SELECTING.value == "selecting"
        assert RoundState.TRAINING.value == "training"
        assert RoundState.AGGREGATING.value == "aggregating"
        assert RoundState.EVALUATING.value == "evaluating"
        assert RoundState.COMPLETED.value == "completed"
        assert RoundState.FAILED.value == "failed"

    def test_terminal_states_are_present(self):
        assert RoundState.COMPLETED is not RoundState.FAILED
        assert RoundState.COMPLETED.value == "completed"
        assert RoundState.FAILED.value == "failed"


# ============================================================
# ClientFailure tests
# ============================================================


class TestClientFailure:
    """Tests for structured client failure records."""

    def test_valid_training_failure(self):
        failure = ClientFailure(
            client_id="client-1",
            phase=RoundState.TRAINING,
            error="training failed",
        )

        assert failure.client_id == "client-1"
        assert failure.phase is RoundState.TRAINING
        assert failure.error == "training failed"

    def test_valid_evaluation_failure(self):
        failure = ClientFailure(
            client_id="client-1",
            phase=RoundState.EVALUATING,
            error="evaluation failed",
        )

        assert failure.phase is RoundState.EVALUATING

    @pytest.mark.parametrize(
        "phase",
        [
            RoundState.CREATED,
            RoundState.SELECTING,
            RoundState.AGGREGATING,
            RoundState.COMPLETED,
            RoundState.FAILED,
        ],
    )
    def test_rejects_invalid_failure_phase(self, phase):
        with pytest.raises(FederatedLearningError):
            ClientFailure(
                client_id="client-1",
                phase=phase,
                error="failure",
            )

    def test_rejects_empty_client_id(self):
        with pytest.raises(FederatedLearningError):
            ClientFailure(
                client_id="",
                phase=RoundState.TRAINING,
                error="failure",
            )

    def test_rejects_empty_error(self):
        with pytest.raises(FederatedLearningError):
            ClientFailure(
                client_id="client-1",
                phase=RoundState.TRAINING,
                error="",
            )

    def test_is_immutable(self):
        failure = ClientFailure(
            client_id="client-1",
            phase=RoundState.TRAINING,
            error="failure",
        )

        with pytest.raises(FrozenInstanceError):
            failure.error = "changed"


# ============================================================
# RoundResult tests
# ============================================================


class TestRoundResult:
    """Tests for immutable round outcomes."""

    def make_fit_result(self):
        return FederatedFitResult(
            client_id="client-1",
            parameters=make_parameters(),
            num_examples=10,
            metrics={"loss": 0.5},
        )

    def test_valid_round_result(self):
        fit_result = self.make_fit_result()

        result = RoundResult(
            round_number=1,
            status=RoundState.COMPLETED,
            selected_clients=("client-1",),
            successful_clients=("client-1",),
            failed_clients=(),
            fit_results={"client-1": fit_result},
            evaluation_results={},
        )

        assert result.round_number == 1
        assert result.status is RoundState.COMPLETED
        assert result.selected_clients == ("client-1",)
        assert result.successful_clients == ("client-1",)
        assert result.failed_clients == ()

    def test_rejects_zero_round_number(self):
        with pytest.raises(FederatedLearningError):
            RoundResult(
                round_number=0,
                status=RoundState.COMPLETED,
                selected_clients=(),
                successful_clients=(),
                failed_clients=(),
                fit_results={},
                evaluation_results={},
            )

    def test_rejects_negative_round_number(self):
        with pytest.raises(FederatedLearningError):
            RoundResult(
                round_number=-1,
                status=RoundState.COMPLETED,
                selected_clients=(),
                successful_clients=(),
                failed_clients=(),
                fit_results={},
                evaluation_results={},
            )

    def test_rejects_duplicate_selected_clients(self):
        with pytest.raises(FederatedLearningError):
            RoundResult(
                round_number=1,
                status=RoundState.COMPLETED,
                selected_clients=("client-1", "client-1"),
                successful_clients=(),
                failed_clients=(),
                fit_results={},
                evaluation_results={},
            )

    def test_rejects_successful_client_not_selected(self):
        fit_result = self.make_fit_result()

        with pytest.raises(FederatedLearningError):
            RoundResult(
                round_number=1,
                status=RoundState.COMPLETED,
                selected_clients=("client-1",),
                successful_clients=("client-2",),
                failed_clients=(),
                fit_results={"client-2": fit_result},
                evaluation_results={},
            )

    def test_rejects_failed_client_not_selected(self):
        failure = ClientFailure(
            client_id="client-2",
            phase=RoundState.TRAINING,
            error="failure",
        )

        with pytest.raises(FederatedLearningError):
            RoundResult(
                round_number=1,
                status=RoundState.FAILED,
                selected_clients=("client-1",),
                successful_clients=(),
                failed_clients=(failure,),
                fit_results={},
                evaluation_results={},
            )

    def test_rejects_client_both_successful_and_failed(self):
        fit_result = self.make_fit_result()

        failure = ClientFailure(
            client_id="client-1",
            phase=RoundState.TRAINING,
            error="failure",
        )

        with pytest.raises(FederatedLearningError):
            RoundResult(
                round_number=1,
                status=RoundState.COMPLETED,
                selected_clients=("client-1",),
                successful_clients=("client-1",),
                failed_clients=(failure,),
                fit_results={"client-1": fit_result},
                evaluation_results={},
            )

    def test_rejects_fit_result_key_mismatch(self):
        fit_result = self.make_fit_result()

        with pytest.raises(FederatedLearningError):
            RoundResult(
                round_number=1,
                status=RoundState.COMPLETED,
                selected_clients=("client-1",),
                successful_clients=("client-1",),
                failed_clients=(),
                fit_results={"client-2": fit_result},
                evaluation_results={},
            )

    def test_result_is_immutable(self):
        result = RoundResult(
            round_number=1,
            status=RoundState.COMPLETED,
            selected_clients=(),
            successful_clients=(),
            failed_clients=(),
            fit_results={},
            evaluation_results={},
        )

        with pytest.raises(FrozenInstanceError):
            result.round_number = 2


# ============================================================
# RoundExecution tests
# ============================================================


class TestRoundExecution:
    """Tests for the server-facing round execution result."""

    def test_copies_aggregated_parameters(self):
        parameters = make_parameters()

        result = RoundResult(
            round_number=1,
            status=RoundState.COMPLETED,
            selected_clients=(),
            successful_clients=(),
            failed_clients=(),
            fit_results={},
            evaluation_results={},
        )

        execution = RoundExecution(
            result=result,
            aggregated_parameters=parameters,
        )

        parameters[0][0, 0] = 999.0

        assert execution.aggregated_parameters[0][0, 0] != 999.0

    def test_rejects_invalid_result(self):
        with pytest.raises(FederatedLearningError):
            RoundExecution(
                result="invalid",
                aggregated_parameters=make_parameters(),
            )

    def test_is_immutable(self):
        result = RoundResult(
            round_number=1,
            status=RoundState.COMPLETED,
            selected_clients=(),
            successful_clients=(),
            failed_clients=(),
            fit_results={},
            evaluation_results={},
        )

        execution = RoundExecution(
            result=result,
            aggregated_parameters=make_parameters(),
        )

        with pytest.raises(FrozenInstanceError):
            execution.result = result


# ============================================================
# RoundCoordinator construction tests
# ============================================================


class TestRoundCoordinatorConstruction:
    """Tests for coordinator dependency validation."""

    def test_requires_strategy(self):
        with pytest.raises(FederatedLearningError):
            RoundCoordinator(
                strategy=None,
                clients={},
            )

    def test_requires_mapping_of_clients(self):
        strategy = FakeStrategy()

        with pytest.raises(FederatedLearningError):
            RoundCoordinator(
                strategy=strategy,
                clients=[],
            )

    def test_requires_at_least_one_client(self):
        strategy = FakeStrategy()

        with pytest.raises(FederatedLearningError):
            RoundCoordinator(
                strategy=strategy,
                clients={},
            )

    def test_rejects_client_id_mismatch(self):
        client = make_client("client-1")
        strategy = FakeStrategy()

        with pytest.raises(FederatedLearningError):
            RoundCoordinator(
                strategy=strategy,
                clients={"wrong-id": client},
            )

    def test_exposes_read_only_client_mapping(self):
        client = make_client("client-1")
        strategy = FakeStrategy()

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={"client-1": client},
        )

        assert coordinator.clients["client-1"] is client

        with pytest.raises(TypeError):
            coordinator.clients["client-2"] = client


# ============================================================
# RoundCoordinator validation tests
# ============================================================


class TestRoundCoordinatorValidation:
    """Tests for execute_round input validation."""

    def test_rejects_zero_round_number(self):
        client = make_client("client-1")
        strategy = FakeStrategy(
            fit_clients=[client],
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={"client-1": client},
        )

        with pytest.raises(FederatedLearningError):
            coordinator.execute_round(
                round_number=0,
                parameters=make_parameters(),
            )

    def test_rejects_negative_round_number(self):
        client = make_client("client-1")
        strategy = FakeStrategy(
            fit_clients=[client],
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={"client-1": client},
        )

        with pytest.raises(FederatedLearningError):
            coordinator.execute_round(
                round_number=-1,
                parameters=make_parameters(),
            )

    def test_rejects_invalid_parameter_payload(self):
        client = make_client("client-1")
        strategy = FakeStrategy(
            fit_clients=[client],
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={"client-1": client},
        )

        with pytest.raises(FederatedLearningError):
            coordinator.execute_round(
                round_number=1,
                parameters=[
                    np.array(
                        [1.0],
                        dtype=np.float64,
                    )
                ],
            )


# ============================================================
# Strategy selection tests
# ============================================================


class TestRoundCoordinatorSelection:
    """Tests for strategy-driven client selection."""

    def test_rejects_empty_selection(self):
        client = make_client("client-1")
        strategy = FakeStrategy(
            fit_clients=[],
        )

        # Force the strategy to explicitly return no clients.
        strategy.select_fit_clients = (
            lambda clients, round_number: []
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={"client-1": client},
        )

        with pytest.raises(FederatedLearningError):
            coordinator.execute_round(
                round_number=1,
                parameters=client.model.get_parameters(),
            )

    def test_rejects_unknown_client_selection(self):
        client = make_client("client-1")
        unknown_client = make_client("client-2")

        strategy = FakeStrategy(
            fit_clients=[unknown_client],
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={"client-1": client},
        )

        with pytest.raises(FederatedLearningError):
            coordinator.execute_round(
                round_number=1,
                parameters=client.model.get_parameters(),
            )

    def test_rejects_duplicate_client_selection(self):
        client = make_client("client-1")

        strategy = FakeStrategy()

        strategy.select_fit_clients = (
            lambda clients, round_number: [
                client,
                client,
            ]
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={"client-1": client},
        )

        with pytest.raises(FederatedLearningError):
            coordinator.execute_round(
                round_number=1,
                parameters=client.model.get_parameters(),
            )


# ============================================================
# Round execution tests
# ============================================================


class TestRoundCoordinatorExecution:
    """
    Tests for the actual federated round execution path.

    These tests use the real FederatedClient abstraction and only
    replace the future Strategy implementation with a controlled
    test double.
    """

    def test_successful_round(self):
        client = make_client("client-1")

        parameters = client.model.get_parameters()

        strategy = FakeStrategy(
            fit_clients=[client],
            aggregated_parameters=[
                parameter.copy()
                for parameter in parameters
            ],
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={"client-1": client},
        )

        execution = coordinator.execute_round(
            round_number=1,
            parameters=parameters,
        )

        assert execution.result.round_number == 1
        assert execution.result.status is RoundState.COMPLETED
        assert execution.result.selected_clients == (
            "client-1",
        )
        assert execution.result.successful_clients == (
            "client-1",
        )
        assert execution.result.failed_clients == ()
        assert "client-1" in execution.result.fit_results

        assert strategy.fit_calls == [1]
        assert strategy.aggregate_calls == [1]

    def test_training_failure_is_recorded(self):
        client = make_client("client-1")

        failing_client = FailingFitClient(
            client_id=client.client_id,
            model=client.model,
            trainer=client.trainer,
            evaluator=client.evaluator,
            parameter_contract=client.parameter_contract,
        )

        parameters = client.model.get_parameters()

        strategy = FakeStrategy(
            fit_clients=[failing_client],
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={"client-1": failing_client},
        )

        with pytest.raises(FederatedLearningError):
            coordinator.execute_round(
                round_number=1,
                parameters=parameters,
            )

    def test_aggregated_parameters_are_returned(self):
        client = make_client("client-1")

        parameters = client.model.get_parameters()

        aggregated = [
            parameter.copy()
            for parameter in parameters
        ]

        strategy = FakeStrategy(
            fit_clients=[client],
            aggregated_parameters=aggregated,
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={"client-1": client},
        )

        execution = coordinator.execute_round(
            round_number=1,
            parameters=parameters,
        )

        assert len(execution.aggregated_parameters) == len(
            aggregated
        )

        for actual, expected in zip(
            execution.aggregated_parameters,
            aggregated,
        ):
            np.testing.assert_array_equal(
                actual,
                expected,
            )

    def test_optional_evaluation_is_skipped_by_default(self):
        client = make_client("client-1")

        parameters = client.model.get_parameters()

        strategy = FakeStrategy(
            fit_clients=[client],
            aggregated_parameters=[
                parameter.copy()
                for parameter in parameters
            ],
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={"client-1": client},
        )

        execution = coordinator.execute_round(
            round_number=1,
            parameters=parameters,
        )

        assert execution.result.evaluation_results == {}
        assert strategy.evaluate_selection_calls == []

    def test_evaluation_selection_is_called_when_requested(self):
        client = make_client("client-1")

        parameters = client.model.get_parameters()

        strategy = FakeStrategy(
            fit_clients=[client],
            evaluate_clients=[client],
            aggregated_parameters=[
                parameter.copy()
                for parameter in parameters
            ],
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={"client-1": client},
        )

        execution = coordinator.execute_round(
            round_number=1,
            parameters=parameters,
            evaluate=True,
        )

        assert strategy.evaluate_selection_calls == [1]
        assert execution.result.status is RoundState.COMPLETED


# ============================================================
# Regression-style contract tests
# ============================================================


class TestRoundContracts:
    """Additional invariants protecting the Phase 3.3-A design."""

    def test_successful_clients_match_fit_result_keys(self):
        fit_result = FederatedFitResult(
            client_id="client-1",
            parameters=make_parameters(),
            num_examples=10,
            metrics={},
        )

        result = RoundResult(
            round_number=1,
            status=RoundState.COMPLETED,
            selected_clients=("client-1",),
            successful_clients=("client-1",),
            failed_clients=(),
            fit_results={"client-1": fit_result},
            evaluation_results={},
        )

        assert set(result.fit_results) == set(
            result.successful_clients
        )

    def test_evaluation_results_can_be_empty(self):
        result = RoundResult(
            round_number=1,
            status=RoundState.COMPLETED,
            selected_clients=("client-1",),
            successful_clients=("client-1",),
            failed_clients=(),
            fit_results={},
            evaluation_results={},
        )

        # This contract test intentionally demonstrates the current
        # relationship between result fields. The coordinator itself
        # always supplies fit results for successful clients.
        assert result.evaluation_results == {}

    def test_round_result_mapping_is_read_only(self):
        result = RoundResult(
            round_number=1,
            status=RoundState.COMPLETED,
            selected_clients=(),
            successful_clients=(),
            failed_clients=(),
            fit_results={},
            evaluation_results={},
        )

        with pytest.raises(TypeError):
            result.fit_results["client-1"] = "invalid"