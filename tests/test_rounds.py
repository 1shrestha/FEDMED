"""
Tests for the FedMed federated round lifecycle and coordinator.

Phase 3.3-A coverage:

- RoundState lifecycle contract
- ClientFailure validation
- RoundResult validation and immutability
- RoundExecution defensive parameter handling
- RoundCoordinator dependency validation
- strategy-driven client selection
- successful federated training round
- client-level training failure
- aggregation result validation
- optional evaluation
- evaluation failure recording
- unknown/duplicate client selection
- invalid parameter payloads
- invalid round numbers

These tests intentionally reuse the established Phase 3.1 and
Phase 3.2 contracts:

    BaseModel
        ↓
    Trainer / Evaluator
        ↓
    FederatedClient
        ↓
    RoundCoordinator

The tests do not modify or weaken those contracts.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
from src.models.base_model import BaseModel
from src.training.evaluator import EvaluationResult, Evaluator
from src.training.trainer import Trainer, TrainingResult


# ============================================================
# Test model
# ============================================================


class TinyFederatedModel(BaseModel):
    """
    Minimal real FedMed BaseModel used by the round tests.

    Using an actual BaseModel is important because Phase 3.1
    ParameterContract.from_model() intentionally rejects arbitrary
    model-like objects.
    """

    def build(self) -> nn.Module:
        return nn.Linear(2, 2)


# ============================================================
# Test Trainer / Evaluator
# ============================================================


class StubTrainer(Trainer):
    """
    Deterministic Trainer test double.

    It inherits from the real Trainer so FederatedClient's type
    boundary remains intact, but overrides train() so the round
    tests do not depend on optimizer behavior.
    """

    def __init__(
        self,
        model: BaseModel,
        *,
        samples_processed: int = 8,
        batches_processed: int = 2,
        epochs_completed: int = 1,
        final_loss: float = 0.25,
    ) -> None:
        # We intentionally do not call Trainer.__init__().
        #
        # FederatedClient only requires that this object is a
        # Trainer and that trainer._model is the same model object.
        self._model = model

        self.samples_processed = samples_processed
        self.batches_processed = batches_processed
        self.epochs_completed = epochs_completed
        self.final_loss = final_loss

        self.train_calls = 0

    def train(
        self,
        dataloader: DataLoader,
        epochs: int | None = None,
    ) -> TrainingResult:
        self.train_calls += 1

        return TrainingResult(
            epochs_completed=self.epochs_completed,
            samples_processed=self.samples_processed,
            batches_processed=self.batches_processed,
            epoch_losses=[self.final_loss],
            final_loss=self.final_loss,
        )


class FailingTrainer(StubTrainer):
    """Trainer that deterministically fails."""

    def train(
        self,
        dataloader: DataLoader,
        epochs: int | None = None,
    ) -> TrainingResult:
        raise FederatedLearningError(
            "synthetic training failure"
        )


class StubEvaluator(Evaluator):
    """
    Deterministic Evaluator test double.

    It preserves the Evaluator type boundary while returning a
    controlled EvaluationResult.
    """

    def __init__(
        self,
        model: BaseModel,
        *,
        samples_evaluated: int = 8,
        batches_evaluated: int = 2,
        loss: float = 0.20,
        metrics: dict[str, float] | None = None,
    ) -> None:
        # As with StubTrainer, bypass the real constructor because
        # the round tests do not need a real criterion/metric stack.
        self._model = model

        self.samples_evaluated = samples_evaluated
        self.batches_evaluated = batches_evaluated
        self.loss = loss
        self.metric_values = metrics or {"accuracy": 0.75}

        self.evaluate_calls = 0

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> EvaluationResult:
        self.evaluate_calls += 1

        return EvaluationResult(
            samples_evaluated=self.samples_evaluated,
            batches_evaluated=self.batches_evaluated,
            loss=self.loss,
            metrics=dict(self.metric_values),
        )


class FailingEvaluator(StubEvaluator):
    """Evaluator that deterministically fails."""

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> EvaluationResult:
        raise FederatedLearningError(
            "synthetic evaluation failure"
        )


# ============================================================
# Strategy test double
# ============================================================


class FakeStrategy:
    """
    Minimal strategy implementation required by RoundCoordinator.

    Phase 3.3-A intentionally uses a private structural strategy
    protocol. The public strategy contract will be finalized in
    Phase 3.3-B.
    """

    def __init__(
        self,
        *,
        fit_clients=None,
        evaluate_clients=None,
        aggregated_parameters=None,
    ) -> None:
        self.fit_clients = fit_clients
        self.evaluate_clients = evaluate_clients

        self.aggregated_parameters = (
            None
            if aggregated_parameters is None
            else [
                parameter.copy()
                for parameter in aggregated_parameters
            ]
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

        if self.fit_clients is not None:
            return list(self.fit_clients)

        return list(clients)

    def aggregate_fit(
        self,
        results,
        round_number: int,
    ):
        self.aggregate_calls.append(round_number)

        if not results:
            raise FederatedLearningError(
                "Synthetic strategy received no results."
            )

        if self.aggregated_parameters is None:
            first_result = next(iter(results.values()))

            return [
                parameter.copy()
                for parameter in first_result.parameters
            ]

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

        if self.evaluate_clients is not None:
            return list(self.evaluate_clients)

        return list(clients)


# ============================================================
# Fixtures / helpers
# ============================================================


def make_dataset(
    size: int = 8,
) -> TensorDataset:
    """Create a tiny deterministic dataset."""

    torch.manual_seed(42)

    samples = torch.randn(size, 2)
    targets = torch.zeros(size, dtype=torch.long)

    return TensorDataset(samples, targets)


def make_loader(
    size: int = 8,
) -> DataLoader:
    """Create a DataLoader accepted by FederatedClient."""

    return DataLoader(
        make_dataset(size),
        batch_size=4,
        shuffle=False,
    )


def make_client(
    client_id: str,
    *,
    trainer_class=StubTrainer,
    evaluator_class=StubEvaluator,
    with_eval_loader: bool = True,
) -> FederatedClient:
    """
    Build a real FederatedClient using a real BaseModel.

    This is the key correction from the previous test suite:
    ParameterContract.from_model() receives an actual BaseModel.
    """

    model = TinyFederatedModel(
        name=f"model-{client_id}",
        device="cpu",
    )

    trainer = trainer_class(model)

    evaluator = evaluator_class(model)

    train_loader = make_loader()

    eval_loader = (
        make_loader()
        if with_eval_loader
        else None
    )

    return FederatedClient(
        client_id=client_id,
        model=model,
        trainer=trainer,
        evaluator=evaluator,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )


def make_parameters(
    client: FederatedClient,
) -> list[np.ndarray]:
    """Return a valid payload for the client's parameter contract."""

    return client.get_parameters()


def make_fit_result(
    client: FederatedClient,
    *,
    value: float | None = None,
) -> FederatedFitResult:
    """
    Construct a valid FederatedFitResult.

    Note that client_id is deliberately NOT passed because the
    real Phase 3.2 FederatedFitResult does not contain that field.
    """

    parameters = client.get_parameters()

    if value is not None:
        parameters = [
            np.full_like(
                parameter,
                value,
            )
            for parameter in parameters
        ]

    return FederatedFitResult(
        parameters=parameters,
        num_examples=8,
        metrics={
            "train_loss": 0.25,
        },
        epochs_completed=1,
        batches_processed=2,
        final_loss=0.25,
    )


def make_evaluate_result() -> FederatedEvaluateResult:
    """Construct a valid federated evaluation result."""

    return FederatedEvaluateResult(
        num_examples=8,
        loss=0.20,
        metrics={
            "accuracy": 0.75,
        },
    )


# ============================================================
# RoundState
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

    def test_terminal_states_are_distinct(self):
        assert RoundState.COMPLETED is not RoundState.FAILED


# ============================================================
# ClientFailure
# ============================================================


class TestClientFailure:
    """Tests for structured client failures."""

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
    def test_rejects_invalid_failure_phase(
        self,
        phase,
    ):
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
# RoundResult
# ============================================================


class TestRoundResult:
    """Tests for immutable round outcomes."""

    def test_valid_round_result(self):
        client = make_client("client-1")

        fit_result = make_fit_result(client)

        result = RoundResult(
            round_number=1,
            status=RoundState.COMPLETED,
            selected_clients=("client-1",),
            successful_clients=("client-1",),
            failed_clients=(),
            fit_results={
                "client-1": fit_result,
            },
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
                selected_clients=(
                    "client-1",
                    "client-1",
                ),
                successful_clients=(),
                failed_clients=(),
                fit_results={},
                evaluation_results={},
            )

    def test_rejects_successful_client_not_selected(self):
        client = make_client("client-2")

        fit_result = make_fit_result(client)

        with pytest.raises(FederatedLearningError):
            RoundResult(
                round_number=1,
                status=RoundState.COMPLETED,
                selected_clients=("client-1",),
                successful_clients=("client-2",),
                failed_clients=(),
                fit_results={
                    "client-2": fit_result,
                },
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
        client = make_client("client-1")

        fit_result = make_fit_result(client)

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
                fit_results={
                    "client-1": fit_result,
                },
                evaluation_results={},
            )

    def test_rejects_fit_result_key_mismatch(self):
        client = make_client("client-1")

        fit_result = make_fit_result(client)

        with pytest.raises(FederatedLearningError):
            RoundResult(
                round_number=1,
                status=RoundState.COMPLETED,
                selected_clients=("client-1",),
                successful_clients=("client-1",),
                failed_clients=(),
                fit_results={
                    "client-2": fit_result,
                },
                evaluation_results={},
            )

    def test_successful_client_requires_fit_result(self):
        """
        Explicitly verify the established invariant:

            successful_clients == fit_results.keys()
        """

        with pytest.raises(FederatedLearningError):
            RoundResult(
                round_number=1,
                status=RoundState.COMPLETED,
                selected_clients=("client-1",),
                successful_clients=("client-1",),
                failed_clients=(),
                fit_results={},
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

    def test_result_mappings_are_read_only(self):
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


# ============================================================
# RoundExecution
# ============================================================


class TestRoundExecution:
    """Tests for the round execution result."""

    def test_copies_aggregated_parameters(self):
        client = make_client("client-1")

        parameters = make_parameters(client)

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

        parameters[0][...] = 999.0

        assert not np.all(
            execution.aggregated_parameters[0] == 999.0
        )

    def test_rejects_invalid_result(self):
        client = make_client("client-1")

        with pytest.raises(FederatedLearningError):
            RoundExecution(
                result="invalid",
                aggregated_parameters=make_parameters(client),
            )

    def test_is_immutable(self):
        client = make_client("client-1")

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
            aggregated_parameters=make_parameters(client),
        )

        with pytest.raises(FrozenInstanceError):
            execution.result = result


# ============================================================
# RoundCoordinator construction
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
                clients={
                    "wrong-id": client,
                },
            )

    def test_exposes_read_only_client_mapping(self):
        client = make_client("client-1")
        strategy = FakeStrategy()

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={
                "client-1": client,
            },
        )

        assert coordinator.clients["client-1"] is client

        with pytest.raises(TypeError):
            coordinator.clients["client-2"] = client


# ============================================================
# RoundCoordinator validation
# ============================================================


class TestRoundCoordinatorValidation:
    """Tests for execute_round input validation."""

    def test_rejects_zero_round_number(self):
        client = make_client("client-1")

        coordinator = RoundCoordinator(
            strategy=FakeStrategy(
                fit_clients=[client],
            ),
            clients={
                "client-1": client,
            },
        )

        with pytest.raises(FederatedLearningError):
            coordinator.execute_round(
                round_number=0,
                parameters=make_parameters(client),
            )

    def test_rejects_negative_round_number(self):
        client = make_client("client-1")

        coordinator = RoundCoordinator(
            strategy=FakeStrategy(
                fit_clients=[client],
            ),
            clients={
                "client-1": client,
            },
        )

        with pytest.raises(FederatedLearningError):
            coordinator.execute_round(
                round_number=-1,
                parameters=make_parameters(client),
            )

    def test_rejects_invalid_parameter_payload(self):
        client = make_client("client-1")

        coordinator = RoundCoordinator(
            strategy=FakeStrategy(
                fit_clients=[client],
            ),
            clients={
                "client-1": client,
            },
        )

        valid_parameters = make_parameters(client)

        invalid_parameters = [
            np.zeros(
                (1,),
                dtype=valid_parameters[0].dtype,
            )
        ]

        with pytest.raises(FederatedLearningError):
            coordinator.execute_round(
                round_number=1,
                parameters=invalid_parameters,
            )


# ============================================================
# Client selection
# ============================================================


class TestRoundCoordinatorSelection:
    """Tests for strategy-driven client selection."""

    def test_rejects_empty_selection(self):
        client = make_client("client-1")

        strategy = FakeStrategy(
            fit_clients=[],
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={
                "client-1": client,
            },
        )

        with pytest.raises(FederatedLearningError):
            coordinator.execute_round(
                round_number=1,
                parameters=make_parameters(client),
            )

    def test_rejects_unknown_client_selection(self):
        client_1 = make_client("client-1")
        client_2 = make_client("client-2")

        strategy = FakeStrategy(
            fit_clients=[client_2],
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={
                "client-1": client_1,
            },
        )

        with pytest.raises(FederatedLearningError):
            coordinator.execute_round(
                round_number=1,
                parameters=make_parameters(client_1),
            )

    def test_rejects_duplicate_client_selection(self):
        client = make_client("client-1")

        strategy = FakeStrategy(
            fit_clients=[
                client,
                client,
            ],
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={
                "client-1": client,
            },
        )

        with pytest.raises(FederatedLearningError):
            coordinator.execute_round(
                round_number=1,
                parameters=make_parameters(client),
            )


# ============================================================
# Successful round
# ============================================================


class TestSuccessfulRound:
    """Tests for the normal round execution path."""

    def test_successful_round(self):
        client = make_client("client-1")

        parameters = make_parameters(client)

        strategy = FakeStrategy(
            fit_clients=[client],
            aggregated_parameters=parameters,
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={
                "client-1": client,
            },
        )

        execution = coordinator.execute_round(
            round_number=1,
            parameters=parameters,
        )

        assert execution.result.round_number == 1

        assert execution.result.status is (
            RoundState.COMPLETED
        )

        assert execution.result.selected_clients == (
            "client-1",
        )

        assert execution.result.successful_clients == (
            "client-1",
        )

        assert execution.result.failed_clients == ()

        assert set(
            execution.result.fit_results
        ) == {"client-1"}

        assert strategy.fit_calls == [1]
        assert strategy.aggregate_calls == [1]

    def test_aggregated_parameters_are_returned(self):
        client = make_client("client-1")

        parameters = make_parameters(client)

        aggregated = [
            parameter + 1.0
            for parameter in parameters
        ]

        strategy = FakeStrategy(
            fit_clients=[client],
            aggregated_parameters=aggregated,
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={
                "client-1": client,
            },
        )

        execution = coordinator.execute_round(
            round_number=1,
            parameters=parameters,
        )

        for actual, expected in zip(
            execution.aggregated_parameters,
            aggregated,
        ):
            np.testing.assert_array_equal(
                actual,
                expected,
            )


# ============================================================
# Client failure behavior
# ============================================================


class TestClientFailures:
    """Tests for expected client-level failures."""

    def test_training_failure_is_recorded(self):
        client = make_client(
            "client-1",
            trainer_class=FailingTrainer,
        )

        parameters = make_parameters(client)

        strategy = FakeStrategy(
            fit_clients=[client],
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={
                "client-1": client,
            },
        )

        with pytest.raises(FederatedLearningError):
            coordinator.execute_round(
                round_number=1,
                parameters=parameters,
            )

    def test_mixed_success_and_failure_is_aggregated(self):
        """
        One successful client and one failed client should still
        permit aggregation because RoundCoordinator currently
        accepts any non-empty successful result set.

        Minimum-client policy belongs to the future strategy/server
        layer, not the local client abstraction.
        """

        successful_client = make_client(
            "client-1",
        )

        failing_client = make_client(
            "client-2",
            trainer_class=FailingTrainer,
        )

        parameters = make_parameters(
            successful_client,
        )

        strategy = FakeStrategy(
            fit_clients=[
                successful_client,
                failing_client,
            ],
            aggregated_parameters=parameters,
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={
                "client-1": successful_client,
                "client-2": failing_client,
            },
        )

        execution = coordinator.execute_round(
            round_number=1,
            parameters=parameters,
        )

        assert execution.result.status is (
            RoundState.COMPLETED
        )

        assert execution.result.selected_clients == (
            "client-1",
            "client-2",
        )

        assert execution.result.successful_clients == (
            "client-1",
        )

        assert len(execution.result.failed_clients) == 1

        failure = execution.result.failed_clients[0]

        assert failure.client_id == "client-2"
        assert failure.phase is RoundState.TRAINING


# ============================================================
# Evaluation
# ============================================================


class TestRoundEvaluation:
    """Tests for optional local evaluation."""

    def test_evaluation_is_skipped_by_default(self):
        client = make_client("client-1")

        parameters = make_parameters(client)

        strategy = FakeStrategy(
            fit_clients=[client],
            aggregated_parameters=parameters,
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={
                "client-1": client,
            },
        )

        execution = coordinator.execute_round(
            round_number=1,
            parameters=parameters,
        )

        assert execution.result.evaluation_results == {}
        assert strategy.evaluate_selection_calls == []

    def test_evaluation_selection_is_called_when_requested(self):
        client = make_client("client-1")

        parameters = make_parameters(client)

        strategy = FakeStrategy(
            fit_clients=[client],
            evaluate_clients=[client],
            aggregated_parameters=parameters,
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={
                "client-1": client,
            },
        )

        execution = coordinator.execute_round(
            round_number=1,
            parameters=parameters,
            evaluate=True,
        )

        assert strategy.evaluate_selection_calls == [1]

        assert execution.result.status is (
            RoundState.COMPLETED
        )

        assert set(
            execution.result.evaluation_results
        ) == {"client-1"}

    def test_evaluation_failure_is_recorded(self):
        client = make_client(
            "client-1",
            evaluator_class=FailingEvaluator,
        )

        parameters = make_parameters(client)

        strategy = FakeStrategy(
            fit_clients=[client],
            evaluate_clients=[client],
            aggregated_parameters=parameters,
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={
                "client-1": client,
            },
        )

        execution = coordinator.execute_round(
            round_number=1,
            parameters=parameters,
            evaluate=True,
        )

        assert execution.result.status is (
            RoundState.COMPLETED
        )

        assert execution.result.evaluation_results == {}

        assert execution.result.failed_clients == ()

        assert len(execution.result.evaluation_failures) == 1

        failure = execution.result.evaluation_failures[0]

        assert failure.client_id == "client-1"
        assert failure.phase is RoundState.EVALUATING


# ============================================================
# Aggregation validation
# ============================================================


class TestAggregationValidation:
    """Tests for validation of strategy aggregation output."""

    def test_rejects_invalid_aggregated_parameter_shape(self):
        client = make_client("client-1")

        parameters = make_parameters(client)

        invalid_aggregated = [
            np.zeros(
                (999,),
                dtype=parameters[0].dtype,
            )
            for _ in parameters
        ]

        strategy = FakeStrategy(
            fit_clients=[client],
            aggregated_parameters=invalid_aggregated,
        )

        coordinator = RoundCoordinator(
            strategy=strategy,
            clients={
                "client-1": client,
            },
        )

        with pytest.raises(FederatedLearningError):
            coordinator.execute_round(
                round_number=1,
                parameters=parameters,
            )


# ============================================================
# Contract regression tests
# ============================================================


class TestRoundContracts:
    """Additional invariants protecting Phase 3.3-A."""

    def test_fit_result_has_no_client_id_field(self):
        """
        Client identity is owned by the RoundCoordinator mapping,
        not duplicated inside FederatedFitResult.

        This protects the Phase 3.2 result contract.
        """

        client = make_client("client-1")

        result = make_fit_result(client)

        assert not hasattr(
            result,
            "client_id",
        )

    def test_successful_clients_match_fit_result_keys(self):
        client = make_client("client-1")

        fit_result = make_fit_result(client)

        result = RoundResult(
            round_number=1,
            status=RoundState.COMPLETED,
            selected_clients=("client-1",),
            successful_clients=("client-1",),
            failed_clients=(),
            fit_results={
                "client-1": fit_result,
            },
            evaluation_results={},
        )

        assert set(
            result.fit_results
        ) == set(
            result.successful_clients
        )

    def test_evaluation_results_can_be_empty(self):
        client = make_client("client-1")

        fit_result = make_fit_result(client)

        result = RoundResult(
            round_number=1,
            status=RoundState.COMPLETED,
            selected_clients=("client-1",),
            successful_clients=("client-1",),
            failed_clients=(),
            fit_results={
                "client-1": fit_result,
            },
            evaluation_results={},
        )

        assert result.evaluation_results == {}

    def test_parameter_contract_matches_client_model(self):
        client = make_client("client-1")

        contract = ParameterContract.from_model(
            client._model
        )

        assert contract.count == (
            client.parameter_contract.count
        )

        assert contract.names == (
            client.parameter_contract.names
        )

        assert contract.shapes == (
            client.parameter_contract.shapes
        )