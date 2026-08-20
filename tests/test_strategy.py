"""
Tests for FedMed Strategy and Strategy/Aggregator integration.

Phase 3.3-B coverage:

- FederatedStrategy abstraction
- FedAvgStrategy construction
- Aggregator dependency injection
- deterministic fit-client selection
- deterministic evaluation-client selection
- round-number validation
- client validation
- duplicate-client protection
- aggregation delegation
- aggregation result propagation
- aggregation failure propagation
- invalid aggregation input handling
- integration with existing RoundCoordinator
- real FederatedClient compatibility
- end-to-end round aggregation

These tests intentionally preserve the existing Phase 3.1,
Phase 3.2, and Phase 3.3-A contracts.

The integration section uses real FedMed FederatedClient,
RoundCoordinator, Trainer/Evaluator type boundaries, and
FedAvgAggregator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.aggregation.fedavg import FedAvgAggregator
from src.common.exceptions import FederatedLearningError
from src.fl.aggregation import Aggregator
from src.fl.client import (
    FederatedClient,
    FederatedFitResult,
)
from src.fl.parameters import ParameterPayload
from src.fl.rounds import (
    RoundCoordinator,
    RoundState,
)
from src.fl.strategy import (
    FedAvgStrategy,
    FederatedStrategy,
)
from src.models.base_model import BaseModel
from src.training.evaluator import EvaluationResult, Evaluator
from src.training.trainer import TrainingResult, Trainer


# ============================================================
# Test model
# ============================================================


class TinyFederatedModel(BaseModel):
    """
    Minimal real FedMed BaseModel.

    Using the real BaseModel boundary is important because
    FederatedClient and ParameterContract intentionally validate
    against BaseModel rather than arbitrary PyTorch modules.
    """

    def build(self) -> nn.Module:
        return nn.Linear(2, 2)


# ============================================================
# Test Trainer / Evaluator
# ============================================================


class StubTrainer(Trainer):
    """
    Deterministic Trainer test double.

    It remains a Trainer instance and exposes the same _model
    reference expected by FederatedClient.
    """

    def __init__(
        self,
        model: BaseModel,
        *,
        value: float = 1.0,
        samples_processed: int = 8,
    ) -> None:
        self._model = model
        self.value = value
        self.samples_processed = samples_processed
        self.train_calls = 0

    def train(
        self,
        dataloader: DataLoader,
        epochs: int | None = None,
    ) -> TrainingResult:
        self.train_calls += 1

        # Deterministically overwrite model state so the integration
        # test produces known client-specific parameters.
        with torch.no_grad():
            for parameter in self._model.network.parameters():
                parameter.fill_(self.value)

        return TrainingResult(
            epochs_completed=1,
            samples_processed=self.samples_processed,
            batches_processed=2,
            epoch_losses=[0.25],
            final_loss=0.25,
        )


class StubEvaluator(Evaluator):
    """Deterministic Evaluator test double."""

    def __init__(
        self,
        model: BaseModel,
    ) -> None:
        self._model = model
        self.evaluate_calls = 0

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> EvaluationResult:
        self.evaluate_calls += 1

        return EvaluationResult(
            samples_evaluated=len(
                dataloader.dataset
            ),
            batches_evaluated=2,
            loss=0.20,
            metrics={
                "accuracy": 0.75,
            },
        )


# ============================================================
# Aggregator test doubles
# ============================================================


class RecordingAggregator(Aggregator):
    """
    Aggregator test double used to prove Strategy delegation.

    It records the exact Mapping supplied by Strategy and returns
    a defensive copy of the first client's parameters.
    """

    def __init__(
        self,
        *,
        output: ParameterPayload | None = None,
    ) -> None:
        self.calls: list[
            dict[str, FederatedFitResult]
        ] = []

        self.output = (
            None
            if output is None
            else [
                parameter.copy()
                for parameter in output
            ]
        )

    def aggregate(
        self,
        results,
    ) -> ParameterPayload:
        self.calls.append(
            dict(results),
        )

        if self.output is not None:
            return [
                parameter.copy()
                for parameter in self.output
            ]

        first_result = next(
            iter(results.values())
        )

        return [
            parameter.copy()
            for parameter in first_result.parameters
        ]


class FailingAggregator(Aggregator):
    """Aggregator that deterministically fails."""

    def aggregate(
        self,
        results,
    ) -> ParameterPayload:
        raise FederatedLearningError(
            "synthetic aggregation failure"
        )


class ExplodingAggregator(Aggregator):
    """Aggregator used to test unexpected exception wrapping."""

    def aggregate(
        self,
        results,
    ) -> ParameterPayload:
        raise RuntimeError(
            "unexpected synthetic failure"
        )


# ============================================================
# Fixtures / helpers
# ============================================================


def make_dataset(
    size: int = 8,
) -> TensorDataset:
    """Create a deterministic dataset."""

    torch.manual_seed(42)

    samples = torch.randn(
        size,
        2,
    )

    targets = torch.zeros(
        size,
        dtype=torch.long,
    )

    return TensorDataset(
        samples,
        targets,
    )


def make_loader(
    size: int = 8,
) -> DataLoader:
    """Create a deterministic DataLoader."""

    return DataLoader(
        make_dataset(size),
        batch_size=4,
        shuffle=False,
    )


def make_client(
    client_id: str,
    *,
    trainer_value: float = 1.0,
    samples_processed: int = 8,
) -> FederatedClient:
    """
    Build a real FedMed FederatedClient.

    This keeps the integration tests aligned with the actual
    Phase 3.2 constructor and model-contract requirements.
    """

    model = TinyFederatedModel(
        name=f"model-{client_id}",
        device="cpu",
    )

    trainer = StubTrainer(
        model,
        value=trainer_value,
        samples_processed=samples_processed,
    )

    evaluator = StubEvaluator(
        model,
    )

    return FederatedClient(
        client_id=client_id,
        model=model,
        trainer=trainer,
        evaluator=evaluator,
        train_loader=make_loader(
            samples_processed,
        ),
        eval_loader=make_loader(
            8,
        ),
    )


def make_fit_result(
    value: float,
    *,
    num_examples: int = 1,
) -> FederatedFitResult:
    """
    Construct a simple synthetic result for Strategy-level tests.

    Strategy does not inspect the parameter mathematics; therefore
    this test helper does not require a real model.
    """

    return FederatedFitResult(
        parameters=[
            np.array(
                [value],
                dtype=np.float32,
            )
        ],
        num_examples=num_examples,
        metrics={
            "train_loss": 0.25,
        },
        epochs_completed=1,
        batches_processed=1,
        final_loss=0.25,
    )


# ============================================================
# FederatedStrategy abstraction
# ============================================================


def test_federated_strategy_is_abstract() -> None:
    """FederatedStrategy cannot be instantiated directly."""

    with pytest.raises(TypeError):
        FederatedStrategy()


def test_fedavg_strategy_is_a_federated_strategy() -> None:
    """FedAvgStrategy must implement the public strategy abstraction."""

    strategy = FedAvgStrategy(
        aggregator=FedAvgAggregator(),
    )

    assert isinstance(
        strategy,
        FederatedStrategy,
    )


# ============================================================
# Construction
# ============================================================


def test_fedavg_strategy_requires_aggregator() -> None:
    """Strategy must receive its aggregation dependency explicitly."""

    with pytest.raises(
        TypeError,
    ):
        # Missing required keyword argument.
        FedAvgStrategy()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "invalid_aggregator",
    [
        None,
        object(),
        "fedavg",
        123,
    ],
)
def test_fedavg_strategy_rejects_invalid_aggregator(
    invalid_aggregator,
) -> None:
    """Only Aggregator implementations may be injected."""

    with pytest.raises(
        FederatedLearningError,
        match="Aggregator",
    ):
        FedAvgStrategy(
            aggregator=invalid_aggregator,
        )


def test_aggregator_property_returns_configured_dependency() -> None:
    """The strategy should expose its configured Aggregator."""

    aggregator = RecordingAggregator()

    strategy = FedAvgStrategy(
        aggregator=aggregator,
    )

    assert strategy.aggregator is aggregator


# ============================================================
# Fit client selection
# ============================================================


def test_fit_selection_returns_all_available_clients() -> None:
    """
    Baseline FedAvg strategy selects every available client.
    """

    clients = [
        make_client("client_a"),
        make_client("client_b"),
        make_client("client_c"),
    ]

    strategy = FedAvgStrategy(
        aggregator=FedAvgAggregator(),
    )

    selected = strategy.select_fit_clients(
        clients,
        round_number=1,
    )

    assert tuple(
        client.client_id
        for client in selected
    ) == (
        "client_a",
        "client_b",
        "client_c",
    )


def test_fit_selection_is_deterministic() -> None:
    """Repeated selection must preserve the available-client order."""

    clients = [
        make_client("client_a"),
        make_client("client_b"),
    ]

    strategy = FedAvgStrategy(
        aggregator=FedAvgAggregator(),
    )

    first = strategy.select_fit_clients(
        clients,
        round_number=1,
    )

    second = strategy.select_fit_clients(
        clients,
        round_number=2,
    )

    assert tuple(
        client.client_id
        for client in first
    ) == tuple(
        client.client_id
        for client in second
    )


def test_fit_selection_does_not_mutate_available_clients() -> None:
    """Strategy must not mutate the coordinator's client collection."""

    clients = [
        make_client("client_a"),
        make_client("client_b"),
    ]

    original = tuple(
        client.client_id
        for client in clients
    )

    strategy = FedAvgStrategy(
        aggregator=FedAvgAggregator(),
    )

    strategy.select_fit_clients(
        clients,
        round_number=1,
    )

    assert tuple(
        client.client_id
        for client in clients
    ) == original


# ============================================================
# Evaluation selection
# ============================================================


def test_evaluation_selection_returns_all_available_clients() -> None:
    """Baseline FedAvg evaluates all available clients."""

    clients = [
        make_client("client_a"),
        make_client("client_b"),
    ]

    strategy = FedAvgStrategy(
        aggregator=FedAvgAggregator(),
    )

    selected = strategy.select_evaluate_clients(
        clients,
        round_number=1,
    )

    assert tuple(
        client.client_id
        for client in selected
    ) == (
        "client_a",
        "client_b",
    )


# ============================================================
# Round validation
# ============================================================


@pytest.mark.parametrize(
    "round_number",
    [
        0,
        -1,
        -100,
        True,
        False,
        1.5,
        "1",
        None,
    ],
)
def test_invalid_round_number_is_rejected(
    round_number,
) -> None:
    """Strategy uses one-based integer round numbers."""

    strategy = FedAvgStrategy(
        aggregator=FedAvgAggregator(),
    )

    clients = [
        make_client("client_a"),
    ]

    with pytest.raises(
        FederatedLearningError,
        match="round_number",
    ):
        strategy.select_fit_clients(
            clients,
            round_number,
        )


# ============================================================
# Client validation
# ============================================================


def test_empty_client_collection_is_rejected() -> None:
    """Baseline strategy cannot select from zero clients."""

    strategy = FedAvgStrategy(
        aggregator=FedAvgAggregator(),
    )

    with pytest.raises(
        FederatedLearningError,
        match="at least one",
    ):
        strategy.select_fit_clients(
            [],
            round_number=1,
        )


def test_invalid_client_object_is_rejected() -> None:
    """Strategy selection requires real FederatedClient objects."""

    strategy = FedAvgStrategy(
        aggregator=FedAvgAggregator(),
    )

    with pytest.raises(
        FederatedLearningError,
        match="FederatedClient",
    ):
        strategy.select_fit_clients(
            [object()],
            round_number=1,
        )


def test_duplicate_client_ids_are_rejected() -> None:
    """
    Duplicate IDs would make the federated participant identity
    ambiguous and must therefore fail.
    """

    client_a1 = make_client("same_id")
    client_a2 = make_client("same_id")

    strategy = FedAvgStrategy(
        aggregator=FedAvgAggregator(),
    )

    with pytest.raises(
        FederatedLearningError,
        match="duplicate",
    ):
        strategy.select_fit_clients(
            [
                client_a1,
                client_a2,
            ],
            round_number=1,
        )


# ============================================================
# Aggregation delegation
# ============================================================


def test_strategy_delegates_aggregation_to_aggregator() -> None:
    """
    The most important Strategy test:

        Strategy.aggregate_fit()
            ->
        Aggregator.aggregate()

    Strategy must not perform parameter arithmetic itself.
    """

    aggregator = RecordingAggregator()

    strategy = FedAvgStrategy(
        aggregator=aggregator,
    )

    results = {
        "client_a": make_fit_result(
            2.0,
            num_examples=100,
        ),
        "client_b": make_fit_result(
            8.0,
            num_examples=300,
        ),
    }

    aggregated = strategy.aggregate_fit(
        results,
        round_number=7,
    )

    assert len(
        aggregator.calls,
    ) == 1

    assert aggregator.calls[0] == results

    np.testing.assert_array_equal(
        aggregated[0],
        np.array(
            [2.0],
            dtype=np.float32,
        ),
    )


def test_strategy_returns_aggregator_output() -> None:
    """Strategy should propagate the Aggregator result."""

    expected = [
        np.array(
            [42.0],
            dtype=np.float32,
        )
    ]

    aggregator = RecordingAggregator(
        output=expected,
    )

    strategy = FedAvgStrategy(
        aggregator=aggregator,
    )

    result = strategy.aggregate_fit(
        {
            "client_a": make_fit_result(
                1.0,
                num_examples=10,
            )
        },
        round_number=1,
    )

    np.testing.assert_array_equal(
        result[0],
        expected[0],
    )


def test_strategy_does_not_replace_aggregator() -> None:
    """The injected Aggregator remains the strategy dependency."""

    aggregator = RecordingAggregator()

    strategy = FedAvgStrategy(
        aggregator=aggregator,
    )

    strategy.aggregate_fit(
        {
            "client_a": make_fit_result(
                1.0,
            )
        },
        round_number=1,
    )

    assert strategy.aggregator is aggregator


def test_fedavg_strategy_performs_actual_fedavg_through_injected_aggregator() -> None:
    """
    Verify the complete Strategy -> FedAvgAggregator path.

    This is deliberately different from the pure Aggregator tests:
    the call starts at Strategy.
    """

    strategy = FedAvgStrategy(
        aggregator=FedAvgAggregator(),
    )

    results = {
        "client_a": make_fit_result(
            2.0,
            num_examples=100,
        ),
        "client_b": make_fit_result(
            6.0,
            num_examples=300,
        ),
    }

    aggregated = strategy.aggregate_fit(
        results,
        round_number=1,
    )

    # 0.25 * 2 + 0.75 * 6 = 5
    np.testing.assert_allclose(
        aggregated[0],
        np.array(
            [5.0],
            dtype=np.float32,
        ),
    )


# ============================================================
# Aggregation failure behavior
# ============================================================


def test_federated_learning_error_from_aggregator_is_preserved() -> None:
    """
    Existing FedMed domain errors should not be unnecessarily
    hidden by Strategy.
    """

    strategy = FedAvgStrategy(
        aggregator=FailingAggregator(),
    )

    with pytest.raises(
        FederatedLearningError,
        match="synthetic aggregation failure",
    ):
        strategy.aggregate_fit(
            {
                "client_a": make_fit_result(
                    1.0,
                )
            },
            round_number=1,
        )


def test_unexpected_aggregator_exception_is_wrapped() -> None:
    """
    Unexpected implementation errors should cross the Strategy
    boundary as a FedMed domain exception.
    """

    strategy = FedAvgStrategy(
        aggregator=ExplodingAggregator(),
    )

    with pytest.raises(
        FederatedLearningError,
        match="Strategy aggregation failed",
    ):
        strategy.aggregate_fit(
            {
                "client_a": make_fit_result(
                    1.0,
                )
            },
            round_number=3,
        )


@pytest.mark.parametrize(
    "results",
    [
        {},
        None,
        [],
        "invalid",
    ],
)
def test_invalid_fit_results_are_rejected(
    results,
) -> None:
    """Strategy must reject invalid aggregation input."""

    strategy = FedAvgStrategy(
        aggregator=RecordingAggregator(),
    )

    with pytest.raises(
        FederatedLearningError,
    ):
        strategy.aggregate_fit(
            results,
            round_number=1,
        )


def test_invalid_fit_result_value_is_rejected() -> None:
    """Strategy must enforce FederatedFitResult values."""

    strategy = FedAvgStrategy(
        aggregator=RecordingAggregator(),
    )

    with pytest.raises(
        FederatedLearningError,
        match="FederatedFitResult",
    ):
        strategy.aggregate_fit(
            {
                "client_a": object(),
            },
            round_number=1,
        )


# ============================================================
# Real RoundCoordinator integration
# ============================================================


def test_strategy_integrates_with_round_coordinator() -> None:
    """
    End-to-end Phase 3 integration:

        RoundCoordinator
              |
              v
        FedAvgStrategy
              |
              v
        FederatedClient.fit()
              |
              v
        FederatedFitResult
              |
              v
        FedAvgAggregator
              |
              v
        ParameterPayload

    This proves the three Phase 3.3-B implementation files fit
    the already-written Phase 3.2/3.3-A boundaries.
    """

    client_a = make_client(
        "client_a",
        trainer_value=1.0,
        samples_processed=4,
    )

    client_b = make_client(
        "client_b",
        trainer_value=3.0,
        samples_processed=12,
    )

    clients = {
        "client_a": client_a,
        "client_b": client_b,
    }

    strategy = FedAvgStrategy(
        aggregator=FedAvgAggregator(),
    )

    coordinator = RoundCoordinator(
        strategy=strategy,
        clients=clients,
    )

    initial_parameters = client_a.get_parameters()

    execution = coordinator.execute_round(
        round_number=1,
        parameters=initial_parameters,
        evaluate=False,
    )

    assert execution.result.status is RoundState.COMPLETED

    assert execution.result.selected_clients == (
        "client_a",
        "client_b",
    )

    assert execution.result.successful_clients == (
        "client_a",
        "client_b",
    )

    assert execution.result.failed_clients == ()

    assert set(
        execution.result.fit_results.keys(),
    ) == {
        "client_a",
        "client_b",
    }

    assert len(
        execution.aggregated_parameters,
    ) == len(
        initial_parameters,
    )

    # The two clients have 4 and 12 examples.
    #
    # Therefore:
    #
    # client_a weight = 4 / 16 = 0.25
    # client_b weight = 12 / 16 = 0.75
    #
    # Their deterministic trainer values are 1 and 3:
    #
    # global = 0.25 * 1 + 0.75 * 3 = 2.5

    for parameter in execution.aggregated_parameters:
        np.testing.assert_allclose(
            parameter,
            np.full_like(
                parameter,
                2.5,
            ),
            rtol=1e-5,
            atol=1e-5,
        )


def test_round_coordinator_uses_strategy_selection() -> None:
    """
    Verify the existing RoundCoordinator actually uses the
    concrete FedAvgStrategy selection boundary.
    """

    client_a = make_client(
        "client_a",
    )

    client_b = make_client(
        "client_b",
    )

    strategy = FedAvgStrategy(
        aggregator=FedAvgAggregator(),
    )

    coordinator = RoundCoordinator(
        strategy=strategy,
        clients={
            "client_a": client_a,
            "client_b": client_b,
        },
    )

    execution = coordinator.execute_round(
        round_number=2,
        parameters=client_a.get_parameters(),
        evaluate=False,
    )

    assert execution.result.selected_clients == (
        "client_a",
        "client_b",
    )


def test_round_coordinator_with_strategy_and_evaluation() -> None:
    """
    Verify Strategy's separate evaluation-selection boundary
    remains compatible with RoundCoordinator.
    """

    client_a = make_client(
        "client_a",
        trainer_value=1.0,
    )

    client_b = make_client(
        "client_b",
        trainer_value=3.0,
    )

    strategy = FedAvgStrategy(
        aggregator=FedAvgAggregator(),
    )

    coordinator = RoundCoordinator(
        strategy=strategy,
        clients={
            "client_a": client_a,
            "client_b": client_b,
        },
    )

    execution = coordinator.execute_round(
        round_number=1,
        parameters=client_a.get_parameters(),
        evaluate=True,
    )

    assert execution.result.status is RoundState.COMPLETED

    assert set(
        execution.result.evaluation_results.keys(),
    ) == {
        "client_a",
        "client_b",
    }

    assert execution.result.evaluation_failures == ()


# ============================================================
# Aggregation contract integration
# ============================================================


def test_strategy_uses_custom_aggregation_algorithm() -> None:
    """
    Prove the Strategy is genuinely decoupled from the mathematical
    aggregation algorithm.

    A custom Aggregator can be injected without modifying Strategy.
    """

    custom_output = [
        np.array(
            [99.0],
            dtype=np.float32,
        )
    ]

    aggregator = RecordingAggregator(
        output=custom_output,
    )

    strategy = FedAvgStrategy(
        aggregator=aggregator,
    )

    result = strategy.aggregate_fit(
        {
            "client_a": make_fit_result(
                1.0,
            ),
            "client_b": make_fit_result(
                2.0,
            ),
        },
        round_number=5,
    )

    np.testing.assert_array_equal(
        result[0],
        custom_output[0],
    )

    assert len(
        aggregator.calls,
    ) == 1