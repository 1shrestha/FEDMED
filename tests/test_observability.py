"""
Tests for structured federated round observability.

Phase 3.5 coverage:

1.  valid round summary creation
2.  round-number validation
3.  inconsistent client-set validation
4.  successful/failed client accounting
5.  metric validation
6.  an empty failed-client set
7.  round history integration
8.  multiple completed rounds
9.  compatibility with existing FederatedServer behavior
10. absence of any Flower dependency in the new core module

The tests deliberately reuse the established contracts:

    BaseModel
        ↓
    Trainer / Evaluator
        ↓
    FederatedClient
        ↓
    RoundCoordinator → RoundExecution
        ↓
    FedAvgStrategy → FedAvgAggregator
        ↓
    FederatedServer

Observability is added on top of those contracts. It does not replace
or weaken any of them.
"""

from __future__ import annotations

import ast
import tokenize
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from src.aggregation.fedavg import FedAvgAggregator
from src.common.config import TrainingConfig
from src.common.exceptions import FederatedLearningError
from src.fl.checkpoint import FederationCheckpointStore
from src.fl.client import (
    FederatedClient,
    FederatedEvaluateResult,
    FederatedFitResult,
)
from src.fl.observability import (
    ACCURACY_METRIC_KEY,
    FINGERPRINT_LENGTH,
    FederatedRoundSummary,
    parameter_fingerprint,
)
from src.fl.parameters import ParameterPayload
from src.fl.rounds import (
    ClientFailure,
    RoundCoordinator,
    RoundExecution,
    RoundResult,
    RoundState,
)
from src.fl.server import FederatedServer
from src.fl.strategy import FedAvgEvaluationResult, FedAvgStrategy
from src.models.base_model import BaseModel
from src.training.evaluator import Evaluator
from src.training.metrics import Accuracy
from src.training.trainer import Trainer


# ======================================================================
# Test model / data
# ======================================================================


class ObservabilityTestModel(BaseModel):
    """Small deterministic model used by the observability tests."""

    def build(self) -> nn.Module:
        return nn.Linear(2, 2)


def make_loader(
    size: int = 8,
) -> DataLoader:
    """
    Create a deterministic DataLoader compatible with the real
    Trainer/Evaluator boundaries.
    """

    torch.manual_seed(42)

    samples = torch.randn(size, 2)
    targets = torch.tensor(
        [0, 1] * (size // 2),
        dtype=torch.long,
    )

    return DataLoader(
        TensorDataset(samples, targets),
        batch_size=4,
        shuffle=False,
    )


# ======================================================================
# Federation helpers
# ======================================================================


def make_client(
    client_id: str,
) -> FederatedClient:
    """
    Construct a real FederatedClient over the real Phase 2
    Trainer/Evaluator and the real BaseModel contract.
    """

    torch.manual_seed(100)

    model = ObservabilityTestModel(
        name=f"observability_model_{client_id}",
        device="cpu",
    )

    criterion = nn.CrossEntropyLoss()

    config = TrainingConfig(
        local_epochs=1,
        batch_size=4,
        learning_rate=0.01,
        optimizer="sgd",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=SGD(
            model.parameters(),
            lr=config.learning_rate,
        ),
        config=config,
    )

    evaluator = Evaluator(
        model=model,
        criterion=criterion,
        metrics=[Accuracy()],
    )

    return FederatedClient(
        client_id=client_id,
        model=model,
        trainer=trainer,
        evaluator=evaluator,
        train_loader=make_loader(8),
        eval_loader=make_loader(8),
    )


def make_clients() -> dict[str, FederatedClient]:
    """Create a deterministic two-client federation."""

    return {
        "client_a": make_client("client_a"),
        "client_b": make_client("client_b"),
    }


def make_strategy() -> FedAvgStrategy:
    """Create the canonical FedAvg strategy."""

    return FedAvgStrategy(
        aggregator=FedAvgAggregator(),
    )


def make_initial_parameters(
    clients: Mapping[str, FederatedClient],
) -> ParameterPayload:
    """Obtain a valid initial payload from a real client contract."""

    return next(iter(clients.values())).get_parameters()


def make_server(
    *,
    clients: Mapping[str, FederatedClient] | None = None,
    strategy: FedAvgStrategy | None = None,
    checkpoint_store: FederationCheckpointStore | None = None,
) -> FederatedServer:
    """Construct a real server over the real round coordinator."""

    if clients is None:
        clients = make_clients()

    if strategy is None:
        strategy = make_strategy()

    return FederatedServer(
        clients=clients,
        strategy=strategy,
        coordinator=RoundCoordinator(
            strategy=strategy,
            clients=clients,
        ),
        initial_parameters=make_initial_parameters(
            clients
        ),
        checkpoint_store=checkpoint_store,
    )


class FailingClient(FederatedClient):
    """FederatedClient whose local training deterministically fails."""

    def fit(
        self,
        parameters: ParameterPayload,
    ):
        raise FederatedLearningError(
            f"Synthetic training failure for client "
            f"'{self.client_id}'."
        )


def make_failing_client(
    client_id: str,
) -> FederatedClient:
    """
    Construct a real FederatedClient and replace only its training
    behavior, preserving the client dependency contract.
    """

    base = make_client(client_id)

    return FailingClient(
        client_id=base.client_id,
        model=base._model,
        trainer=base._trainer,
        evaluator=base._evaluator,
        train_loader=base._train_loader,
        eval_loader=base._eval_loader,
    )


# ======================================================================
# Direct summary construction
# ======================================================================


def make_summary(**overrides) -> FederatedRoundSummary:
    """Build a valid summary, overriding individual fields."""

    fields_ = {
        "round_number": 1,
        "status": RoundState.COMPLETED,
        "selected_clients": ("client_a", "client_b"),
        "successful_clients": ("client_a", "client_b"),
        "failed_clients": (),
        "evaluation_failures": (),
        "training_examples": 16,
        "training_loss": 0.25,
        "evaluation_examples": 0,
        "evaluation_loss": None,
        "evaluation_accuracy": None,
        "parameter_fingerprint": "0123456789abcdef",
    }

    fields_.update(overrides)

    return FederatedRoundSummary(**fields_)


# ======================================================================
# RoundExecution construction
# ======================================================================


def make_fit_result(
    client: FederatedClient,
    *,
    num_examples: int = 8,
    final_loss: float = 0.25,
) -> FederatedFitResult:
    """Construct a valid FederatedFitResult for a real client."""

    return FederatedFitResult(
        parameters=client.get_parameters(),
        num_examples=num_examples,
        metrics={"train_loss": final_loss},
        epochs_completed=1,
        batches_processed=2,
        final_loss=final_loss,
    )


def make_execution(
    *,
    round_number: int = 1,
    clients: Mapping[str, FederatedClient] | None = None,
    failed_clients: tuple[ClientFailure, ...] = (),
    evaluation_results: (
        Mapping[str, FederatedEvaluateResult] | None
    ) = None,
    evaluation_failures: tuple[ClientFailure, ...] = (),
    fit_losses: Mapping[str, float] | None = None,
) -> RoundExecution:
    """Build a valid RoundExecution for derivation tests."""

    if clients is None:
        clients = {"client_a": make_client("client_a")}

    successful = tuple(clients)

    fit_results = {
        client_id: make_fit_result(
            clients[client_id],
            final_loss=(fit_losses or {}).get(
                client_id, 0.25
            ),
        )
        for client_id in successful
    }

    selected = successful + tuple(
        failure.client_id
        for failure in failed_clients
    )

    result = RoundResult(
        round_number=round_number,
        status=RoundState.COMPLETED,
        selected_clients=selected,
        successful_clients=successful,
        failed_clients=failed_clients,
        fit_results=fit_results,
        evaluation_results=(
            evaluation_results or {}
        ),
        evaluation_failures=evaluation_failures,
    )

    return RoundExecution(
        result=result,
        aggregated_parameters=next(
            iter(clients.values())
        ).get_parameters(),
    )


def make_dropout_failure(
    client_id: str = "client_b",
) -> ClientFailure:
    """Return a structured TRAINING failure."""

    return ClientFailure(
        client_id=client_id,
        phase=RoundState.TRAINING,
        error="controlled dropout",
    )


def make_evaluation_failure(
    client_id: str = "client_b",
) -> ClientFailure:
    """Return a structured EVALUATING failure."""

    return ClientFailure(
        client_id=client_id,
        phase=RoundState.EVALUATING,
        error="controlled evaluation failure",
    )


# ======================================================================
# 1. Valid round summary creation
# ======================================================================


class TestValidSummaryCreation:
    """A valid summary records exactly what it was given."""

    def test_records_all_fields(self):
        summary = make_summary()

        assert summary.round_number == 1
        assert summary.status is RoundState.COMPLETED
        assert summary.selected_clients == (
            "client_a",
            "client_b",
        )
        assert summary.successful_clients == (
            "client_a",
            "client_b",
        )
        assert summary.failed_clients == ()
        assert summary.evaluation_failures == ()
        assert summary.training_examples == 16
        assert summary.training_loss == pytest.approx(0.25)
        assert summary.evaluation_examples == 0
        assert summary.evaluation_loss is None
        assert summary.evaluation_accuracy is None
        assert (
            summary.parameter_fingerprint
            == "0123456789abcdef"
        )

    def test_normalizes_sequences_to_tuples(self):
        summary = make_summary(
            selected_clients=["client_a"],
            successful_clients=["client_a"],
            training_examples=8,
        )

        assert isinstance(
            summary.selected_clients, tuple
        )
        assert isinstance(
            summary.successful_clients, tuple
        )

    def test_is_immutable(self):
        summary = make_summary()

        with pytest.raises(FrozenInstanceError):
            summary.round_number = 2

    def test_fingerprint_is_optional(self):
        summary = make_summary(
            parameter_fingerprint=None,
        )

        assert summary.parameter_fingerprint is None

    def test_evaluation_observations_are_recorded(self):
        summary = make_summary(
            evaluation_examples=16,
            evaluation_loss=0.20,
            evaluation_accuracy=0.75,
        )

        assert summary.evaluated is True
        assert summary.evaluation_loss == pytest.approx(0.20)
        assert summary.evaluation_accuracy == pytest.approx(
            0.75
        )

    def test_derived_participation_views(self):
        summary = make_summary(
            successful_clients=("client_a",),
            failed_clients=(
                make_dropout_failure(),
            ),
        )

        assert summary.participation_count == 1
        assert summary.dropout_count == 1
        assert summary.failed_client_ids == ("client_b",)
        assert (
            summary.evaluation_failed_client_ids == ()
        )


# ======================================================================
# 2. Round-number validation
# ======================================================================


class TestRoundNumberValidation:
    """FedMed rounds are one-based integers."""

    def test_rejects_zero_round(self):
        with pytest.raises(FederatedLearningError):
            make_summary(round_number=0)

    def test_rejects_negative_round(self):
        with pytest.raises(FederatedLearningError):
            make_summary(round_number=-1)

    @pytest.mark.parametrize(
        "round_number",
        [1.0, "1", None],
    )
    def test_rejects_non_integer_round(
        self,
        round_number,
    ):
        with pytest.raises(FederatedLearningError):
            make_summary(round_number=round_number)

    def test_rejects_bool_round(self):
        # bool is a subclass of int, so it needs its own guard.
        with pytest.raises(FederatedLearningError):
            make_summary(round_number=True)


# ======================================================================
# 3. Inconsistent client-set validation
# ======================================================================


class TestClientSetValidation:
    """Client accounting must remain internally consistent."""

    def test_rejects_successful_client_not_selected(self):
        with pytest.raises(FederatedLearningError):
            make_summary(
                selected_clients=("client_a",),
                successful_clients=(
                    "client_a",
                    "client_b",
                ),
                training_examples=16,
            )

    def test_rejects_duplicate_selected_clients(self):
        with pytest.raises(FederatedLearningError):
            make_summary(
                selected_clients=(
                    "client_a",
                    "client_a",
                ),
            )

    def test_rejects_duplicate_successful_clients(self):
        with pytest.raises(FederatedLearningError):
            make_summary(
                successful_clients=(
                    "client_a",
                    "client_a",
                ),
                training_examples=16,
            )

    def test_rejects_empty_client_id(self):
        with pytest.raises(FederatedLearningError):
            make_summary(
                selected_clients=("client_a", ""),
                successful_clients=("client_a",),
                training_examples=8,
            )

    def test_rejects_non_string_client_id(self):
        with pytest.raises(FederatedLearningError):
            make_summary(
                selected_clients=("client_a", 7),
                successful_clients=("client_a",),
                training_examples=8,
            )

    def test_rejects_failed_client_not_selected(self):
        with pytest.raises(FederatedLearningError):
            make_summary(
                selected_clients=("client_a",),
                successful_clients=("client_a",),
                failed_clients=(
                    make_dropout_failure("client_b"),
                ),
                training_examples=8,
            )

    def test_rejects_training_failure_phase_in_failures(self):
        with pytest.raises(FederatedLearningError):
            make_summary(
                failed_clients=(
                    make_evaluation_failure("client_b"),
                ),
                successful_clients=("client_a",),
                training_examples=8,
            )

    def test_rejects_training_phase_in_evaluation_failures(
        self,
    ):
        with pytest.raises(FederatedLearningError):
            make_summary(
                evaluation_failures=(
                    make_dropout_failure("client_a"),
                ),
            )

    def test_rejects_evaluation_failure_not_selected(self):
        with pytest.raises(FederatedLearningError):
            make_summary(
                selected_clients=("client_a",),
                successful_clients=("client_a",),
                evaluation_failures=(
                    make_evaluation_failure("client_b"),
                ),
                training_examples=8,
            )

    def test_rejects_non_client_failure_object(self):
        with pytest.raises(FederatedLearningError):
            make_summary(
                failed_clients=("client_b",),
                successful_clients=("client_a",),
                training_examples=8,
            )

    def test_rejects_invalid_status(self):
        with pytest.raises(FederatedLearningError):
            make_summary(status="completed")

    def test_rejects_duplicate_failure_ids(self):
        failure = make_dropout_failure()

        with pytest.raises(FederatedLearningError):
            make_summary(
                successful_clients=("client_a",),
                training_examples=8,
                failed_clients=(failure, failure),
            )

    def test_summary_survives_the_same_client_set_as_round_result(
        self,
    ):
        # The summary reuses the RoundResult vocabulary, so the two
        # must accept exactly the same client sets.
        clients = {
            "client_a": make_client("client_a"),
        }

        execution = make_execution(
            clients=clients,
            failed_clients=(
                make_dropout_failure("client_b"),
            ),
        )

        result = execution.result

        summary = (
            FederatedRoundSummary.from_round_execution(
                execution,
            )
        )

        assert (
            summary.selected_clients
            == result.selected_clients
        )
        assert (
            summary.successful_clients
            == result.successful_clients
        )


# ======================================================================
# 4. Successful / failed client accounting
# ======================================================================


class TestClientFailureAccounting:
    """
    A failed client is never reported as successful, even though the
    round itself completed.
    """

    def test_failed_client_is_not_successful(self):
        with pytest.raises(FederatedLearningError):
            make_summary(
                failed_clients=(
                    make_dropout_failure(),
                ),
                training_examples=16,
            )

    def test_completed_round_with_dropout(self):
        # This is the E6-shaped case: the round completes, but only
        # one of the two selected clients is counted as successful.
        summary = make_summary(
            successful_clients=("client_a",),
            failed_clients=(
                make_dropout_failure(),
            ),
            training_examples=8,
        )

        assert summary.status is RoundState.COMPLETED
        assert summary.successful_clients == ("client_a",)
        assert summary.failed_client_ids == ("client_b",)
        assert summary.participation_count == 1
        assert summary.dropout_count == 1

        assert set(
            summary.successful_clients
        ).isdisjoint(summary.failed_client_ids)

    def test_evaluation_failure_does_not_remove_training_success(
        self,
    ):
        # A client that trained and then failed evaluation is still a
        # successful training client. This mirrors RoundResult.
        summary = make_summary(
            evaluation_failures=(
                make_evaluation_failure(),
            ),
        )

        assert summary.successful_clients == (
            "client_a",
            "client_b",
        )
        assert (
            summary.evaluation_failed_client_ids
            == ("client_b",)
        )
        assert summary.dropout_count == 0

    def test_training_examples_must_be_zero_without_success(self):
        with pytest.raises(FederatedLearningError):
            make_summary(
                successful_clients=(),
                training_examples=8,
                training_loss=None,
            )

    def test_training_loss_required_with_success(self):
        with pytest.raises(FederatedLearningError):
            make_summary(
                successful_clients=("client_a",),
                training_examples=8,
                training_loss=None,
            )

    def test_training_loss_forbidden_without_success(self):
        with pytest.raises(FederatedLearningError):
            make_summary(
                successful_clients=(),
                training_examples=0,
                training_loss=0.25,
            )


# ======================================================================
# 5. Metric validation
# ======================================================================


class TestMetricValidation:
    """Recorded metrics must be finite and non-negative."""

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf")],
    )
    def test_rejects_non_finite_training_loss(
        self,
        value,
    ):
        with pytest.raises(FederatedLearningError):
            make_summary(training_loss=value)

    def test_rejects_negative_training_loss(self):
        with pytest.raises(FederatedLearningError):
            make_summary(training_loss=-0.1)

    def test_rejects_non_numeric_training_loss(self):
        with pytest.raises(FederatedLearningError):
            make_summary(training_loss="0.25")

    def test_rejects_non_finite_evaluation_loss(self):
        with pytest.raises(FederatedLearningError):
            make_summary(
                evaluation_examples=8,
                evaluation_loss=float("nan"),
            )

    @pytest.mark.parametrize(
        "value",
        [-0.01, 1.01, 2.0],
    )
    def test_rejects_accuracy_outside_unit_interval(
        self,
        value,
    ):
        with pytest.raises(FederatedLearningError):
            make_summary(
                evaluation_examples=8,
                evaluation_loss=0.2,
                evaluation_accuracy=value,
            )

    @pytest.mark.parametrize(
        "value",
        [0.0, 0.5, 1.0],
    )
    def test_accepts_accuracy_inside_unit_interval(
        self,
        value,
    ):
        summary = make_summary(
            evaluation_examples=8,
            evaluation_loss=0.2,
            evaluation_accuracy=value,
        )

        assert summary.evaluation_accuracy == pytest.approx(
            value
        )

    def test_rejects_negative_training_examples(self):
        with pytest.raises(FederatedLearningError):
            make_summary(training_examples=-1)

    def test_rejects_non_integer_training_examples(self):
        with pytest.raises(FederatedLearningError):
            make_summary(training_examples=8.5)

    def test_rejects_metrics_without_evaluation_examples(self):
        with pytest.raises(FederatedLearningError):
            make_summary(
                evaluation_examples=0,
                evaluation_loss=0.2,
            )

    def test_requires_a_metric_when_examples_present(self):
        with pytest.raises(FederatedLearningError):
            make_summary(evaluation_examples=8)

    def test_rejects_empty_fingerprint(self):
        with pytest.raises(FederatedLearningError):
            make_summary(parameter_fingerprint="  ")

    def test_rejects_non_string_fingerprint(self):
        with pytest.raises(FederatedLearningError):
            make_summary(parameter_fingerprint=123)


# ======================================================================
# 6. Empty failed-client set
# ======================================================================


class TestEmptyFailureSet:
    """A clean round reports no failures at all."""

    def test_empty_failed_client_set(self):
        summary = make_summary(
            failed_clients=(),
            evaluation_failures=(),
        )

        assert summary.failed_clients == ()
        assert summary.failed_client_ids == ()
        assert summary.dropout_count == 0
        assert summary.participation_count == 2

    def test_successful_clients_equal_selected_clients(self):
        summary = make_summary()

        assert (
            summary.successful_clients
            == summary.selected_clients
        )

    def test_empty_failure_set_from_real_round(self):
        summary = (
            FederatedRoundSummary.from_round_execution(
                make_execution(),
            )
        )

        assert summary.failed_clients == ()
        assert summary.evaluation_failures == ()
        assert summary.failed_client_ids == ()
        assert summary.dropout_count == 0


# ======================================================================
# 7. Round history integration
# ======================================================================


class TestRoundHistoryIntegration:
    """Summaries are derived from the real round lifecycle."""

    def test_summary_derived_from_execution(self):
        execution = make_execution()

        summary = (
            FederatedRoundSummary.from_round_execution(
                execution,
            )
        )

        assert summary.round_number == 1
        assert summary.status is RoundState.COMPLETED
        assert summary.successful_clients == ("client_a",)
        assert summary.training_examples == 8
        assert summary.training_loss == pytest.approx(0.25)
        assert summary.parameter_fingerprint == (
            parameter_fingerprint(
                execution.aggregated_parameters,
            )
        )

    def test_training_loss_is_sample_weighted(self):
        clients = {
            "client_a": make_client("client_a"),
            "client_b": make_client("client_b"),
        }

        execution = make_execution(
            clients=clients,
            fit_losses={
                "client_a": 0.5,
                "client_b": 0.25,
            },
        )

        summary = (
            FederatedRoundSummary.from_round_execution(
                execution,
            )
        )

        assert summary.training_examples == 16
        assert summary.training_loss == pytest.approx(
            0.375
        )

    def test_derivation_preserves_dropout_accounting(self):
        execution = make_execution(
            failed_clients=(
                make_dropout_failure(),
            ),
        )

        summary = (
            FederatedRoundSummary.from_round_execution(
                execution,
            )
        )

        assert summary.successful_clients == ("client_a",)
        assert summary.failed_client_ids == ("client_b",)

        # Only the successful client's examples are counted.
        assert summary.training_examples == 8

    def test_derivation_records_evaluation_observations(self):
        execution = make_execution(
            evaluation_results={
                "client_a": FederatedEvaluateResult(
                    num_examples=8,
                    loss=0.20,
                    metrics={
                        ACCURACY_METRIC_KEY: 0.75,
                    },
                ),
            },
        )

        summary = (
            FederatedRoundSummary.from_round_execution(
                execution,
            )
        )

        assert summary.evaluation_examples == 8
        assert summary.evaluation_loss == pytest.approx(0.20)
        assert summary.evaluation_accuracy == pytest.approx(
            0.75
        )

    def test_supplied_strategy_aggregate_is_used_verbatim(
        self,
    ):
        execution = make_execution(
            evaluation_results={
                "client_a": FederatedEvaluateResult(
                    num_examples=8,
                    loss=0.20,
                    metrics={
                        ACCURACY_METRIC_KEY: 0.75,
                    },
                ),
            },
        )

        summary = (
            FederatedRoundSummary.from_round_execution(
                execution,
                evaluation=FedAvgEvaluationResult(
                    loss=0.31,
                    metrics={
                        ACCURACY_METRIC_KEY: 0.90,
                    },
                ),
            )
        )

        assert summary.evaluation_loss == pytest.approx(0.31)
        assert summary.evaluation_accuracy == pytest.approx(
            0.90
        )

    def test_supplied_aggregate_without_evaluation_is_rejected(
        self,
    ):
        with pytest.raises(FederatedLearningError):
            FederatedRoundSummary.from_round_execution(
                make_execution(),
                evaluation=FedAvgEvaluationResult(
                    loss=0.2,
                    metrics={},
                ),
            )

    def test_rejects_invalid_evaluation_type(self):
        execution = make_execution(
            evaluation_results={
                "client_a": FederatedEvaluateResult(
                    num_examples=8,
                    loss=0.2,
                    metrics={},
                ),
            },
        )

        with pytest.raises(FederatedLearningError):
            FederatedRoundSummary.from_round_execution(
                execution,
                evaluation="not-an-aggregate",
            )

    def test_rejects_invalid_execution(self):
        with pytest.raises(FederatedLearningError):
            FederatedRoundSummary.from_round_execution(
                "not-an-execution",
            )

    def test_fingerprint_can_be_omitted(self):
        summary = (
            FederatedRoundSummary.from_round_execution(
                make_execution(),
                include_fingerprint=False,
            )
        )

        assert summary.parameter_fingerprint is None

    def test_summary_holds_no_heavyweight_objects(self):
        summary = (
            FederatedRoundSummary.from_round_execution(
                make_execution(
                    failed_clients=(
                        make_dropout_failure(),
                    ),
                ),
            )
        )

        for value in vars(summary).values():

            assert not isinstance(
                value,
                (
                    FederatedFitResult,
                    FederatedEvaluateResult,
                    np.ndarray,
                    list,
                    dict,
                ),
            )

            if isinstance(value, tuple):
                for item in value:
                    assert isinstance(
                        item, (str, ClientFailure)
                    )

        # Every declared field is a scalar or an immutable tuple,
        # which is what keeps the record cheap to retain and easy to
        # serialize later.
        declared_types = {
            declared.name: declared.type
            for declared in fields(summary)
        }

        assert declared_types == {
            "round_number": "int",
            "status": "RoundState",
            "selected_clients": "tuple[str, ...]",
            "successful_clients": "tuple[str, ...]",
            "failed_clients": "tuple[ClientFailure, ...]",
            "evaluation_failures": "tuple[ClientFailure, ...]",
            "training_examples": "int",
            "training_loss": "float | None",
            "evaluation_examples": "int",
            "evaluation_loss": "float | None",
            "evaluation_accuracy": "float | None",
            "parameter_fingerprint": "str | None",
        }

    def test_server_exposes_summaries_after_a_round(self):
        server = make_server()

        assert server.round_summaries == ()

        execution = server.run_round()

        summaries = server.round_summaries

        assert len(summaries) == 1

        summary = summaries[0]

        assert isinstance(
            summary, FederatedRoundSummary
        )
        assert summary.round_number == 1
        assert summary.status is RoundState.COMPLETED
        assert (
            summary.round_number
            == execution.result.round_number
        )
        assert summary.successful_clients == (
            "client_a",
            "client_b",
        )
        assert summary.training_examples == 16
        assert summary.training_loss is not None
        assert summary.parameter_fingerprint is not None

    def test_server_summaries_match_round_history(self):
        server = make_server()

        server.run_round()

        assert tuple(
            summary.round_number
            for summary in server.round_summaries
        ) == tuple(
            execution.result.round_number
            for execution in server.round_history
        )

    def test_server_round_summaries_are_read_only_views(self):
        server = make_server()

        server.run_round()

        first = server.round_summaries

        server.run_round()

        # The earlier tuple is unaffected by a later round.
        assert len(first) == 1
        assert len(server.round_summaries) == 2

    def test_summary_for_round_lookup(self):
        server = make_server()

        server.run_round()

        assert server.summary_for_round(1).round_number == 1

    def test_summary_for_round_rejects_missing_round(self):
        server = make_server()

        server.run_round()

        with pytest.raises(FederatedLearningError):
            server.summary_for_round(7)

    def test_summary_for_round_rejects_invalid_round(self):
        with pytest.raises(FederatedLearningError):
            make_server().summary_for_round("1")

    def test_summary_for_round_rejects_bool_round(self):
        with pytest.raises(FederatedLearningError):
            make_server().summary_for_round(True)


# ======================================================================
# 8. Multiple completed rounds
# ======================================================================


class TestMultipleCompletedRounds:
    """Summaries stay ordered and per-round across a federation."""

    def test_summaries_across_three_rounds(self):
        server = make_server()

        server.run(num_rounds=3)

        summaries = server.round_summaries

        assert [
            summary.round_number for summary in summaries
        ] == [1, 2, 3]

        assert all(
            isinstance(
                summary,
                FederatedRoundSummary,
            )
            for summary in summaries
        )

        assert all(
            summary.status is RoundState.COMPLETED
            for summary in summaries
        )

        assert all(
            summary.training_examples == 16
            for summary in summaries
        )

    def test_summaries_record_evaluation_when_requested(self):
        server = make_server()

        server.run_round(evaluate=True)

        summary = server.round_summaries[0]

        assert summary.evaluated is True
        assert summary.evaluation_examples == 16
        assert summary.evaluation_loss is not None
        assert summary.evaluation_accuracy is not None

    def test_summaries_absent_without_evaluation(self):
        server = make_server()

        server.run_round()

        summary = server.round_summaries[0]

        assert summary.evaluated is False
        assert summary.evaluation_examples == 0
        assert summary.evaluation_loss is None
        assert summary.evaluation_accuracy is None

    def test_fingerprint_reflects_changing_parameters(self):
        server = make_server()

        server.run(num_rounds=2)

        fingerprints = [
            summary.parameter_fingerprint
            for summary in server.round_summaries
        ]

        assert fingerprints[0] != fingerprints[1]

    def test_summaries_agree_with_history_length(self):
        server = make_server()

        server.run(num_rounds=2)

        assert len(server.round_summaries) == len(
            server.round_history
        )
        assert server.completed_round == 2


# ======================================================================
# 9. FederatedServer compatibility
# ======================================================================


class TestServerCompatibility:
    """Observability does not change existing server behavior."""

    def test_server_state_and_history_unchanged(self):
        server = make_server()

        execution = server.run_round()

        assert server.completed_round == 1
        assert server.round_history == (execution,)

    def test_summaries_reflect_dropout_without_changing_policy(
        self,
    ):
        clients = {
            "client_a": make_client("client_a"),
            "client_b": make_failing_client("client_b"),
        }

        server = make_server(clients=clients)

        execution = server.run_round()

        # The round still completes, exactly as before.
        assert execution.result.status is RoundState.COMPLETED
        assert server.completed_round == 1

        summary = server.round_summaries[0]

        assert summary.successful_clients == ("client_a",)
        assert summary.failed_client_ids == ("client_b",)
        assert summary.dropout_count == 1

        # Only the successful client's examples are counted.
        assert summary.training_examples == 8
        assert summary.participation_count == 1

    def test_failed_round_records_no_summary(self):
        clients = {
            "client_a": make_failing_client("client_a"),
        }

        server = make_server(clients=clients)

        with pytest.raises(FederatedLearningError):
            server.run_round()

        assert server.round_summaries == ()
        assert server.round_history == ()
        assert server.completed_round == 0

    def test_checkpoint_restore_does_not_restore_summaries(
        self,
        tmp_path: Path,
    ):
        store = FederationCheckpointStore(
            tmp_path / "checkpoints",
            save_every_round=1,
        )

        source = make_server(checkpoint_store=store)

        source.run(num_rounds=2)

        assert len(source.round_summaries) == 2

        # A brand-new server, as after a process restart.
        resumed = make_server(checkpoint_store=store)

        resumed.restore_checkpoint()

        assert resumed.completed_round == 2
        assert resumed.round_summaries == ()
        assert resumed.round_history == ()

        # The resumed server keeps numbering contiguously.
        resumed.run_round()

        assert [
            summary.round_number
            for summary in resumed.round_summaries
        ] == [3]

    def test_fingerprint_matches_runtime_adapter(self):
        # The Flower adapter and the core observability layer must
        # report the same fingerprint for the same payload.
        from app.server import FedMedFlowerStrategy

        parameters = make_client("client_a").get_parameters()

        assert (
            FedMedFlowerStrategy._parameter_fingerprint(
                parameters
            )
            == parameter_fingerprint(parameters)
        )


# ======================================================================
# 10. Framework independence
# ======================================================================


def observability_module_path() -> Path:
    """Return the path of the new core observability module."""

    return (
        Path(__file__).resolve().parents[1]
        / "src"
        / "fl"
        / "observability.py"
    )


def module_code_only(path: Path) -> str:
    """
    Return a module's source with comments and string literals
    removed.

    Documentation may legitimately name a framework it deliberately
    avoids importing, so the dependency scan must look at real code
    rather than at prose.
    """

    kept: list[str] = []

    with path.open(encoding="utf-8") as handle:

        for token in tokenize.generate_tokens(
            handle.readline
        ):
            if token.type in {
                tokenize.COMMENT,
                tokenize.STRING,
            }:
                continue

            kept.append(token.string)

    return " ".join(kept)


class TestFrameworkIndependence:
    """The new core module must be Flower-free."""

    def test_code_mentions_no_flower_api(self):
        code = module_code_only(observability_module_path())

        for forbidden in (
            "import flwr",
            "from flwr",
            "ServerApp",
            "ClientApp",
            "MetricRecord",
            "ArrayRecord",
        ):
            assert forbidden not in code

    def test_source_imports_no_flower_module(self):
        import src.fl.observability as module

        tree = ast.parse(
            Path(module.__file__).read_text(
                encoding="utf-8"
            )
        )

        imported: set[str] = set()

        for node in ast.walk(tree):

            if isinstance(node, ast.Import):
                imported.update(
                    alias.name
                    for alias in node.names
                )

            elif isinstance(
                node,
                ast.ImportFrom,
            ):
                imported.add(node.module or "")

        assert not any(
            name.split(".")[0] == "flwr"
            for name in imported
        )

    def test_module_has_no_flower_attribute(self):
        import src.fl.observability as module

        assert not hasattr(module, "flwr")

    def test_parameter_fingerprint_is_deterministic(self):
        parameters = make_client("client_a").get_parameters()

        first = parameter_fingerprint(parameters)
        second = parameter_fingerprint(
            [array.copy() for array in parameters]
        )

        assert first == second
        assert len(first) == FINGERPRINT_LENGTH

    def test_parameter_fingerprint_detects_change(self):
        parameters = make_client("client_a").get_parameters()

        baseline = parameter_fingerprint(parameters)

        changed = [array.copy() for array in parameters]
        changed[0] = changed[0] + 1.0

        assert parameter_fingerprint(changed) != baseline

    def test_parameter_fingerprint_is_order_sensitive(self):
        parameters = make_client("client_a").get_parameters()

        reversed_payload = list(reversed(parameters))

        assert parameter_fingerprint(
            reversed_payload
        ) != parameter_fingerprint(parameters)

    def test_parameter_fingerprint_rejects_string(self):
        with pytest.raises(FederatedLearningError):
            parameter_fingerprint("not-a-payload")

    def test_parameter_fingerprint_rejects_non_sequence(self):
        with pytest.raises(FederatedLearningError):
            parameter_fingerprint(
                np.zeros((2, 2))
            )

    def test_parameter_fingerprint_accepts_numpy_payload(self):
        payload = [
            np.asarray(array)
            for array in make_client("client_a")
            .get_parameters()
        ]

        assert parameter_fingerprint(payload)
