"""
Tests for the FedMed Phase 3.2 federated client.

Covers:

- Client construction and dependency validation
- Client/model/Trainer/Evaluator consistency
- Parameter extraction and loading
- Parameter contract enforcement
- Defensive parameter handling
- Local training orchestration
- Local evaluation orchestration
- Sample-count propagation
- Metric propagation
- Failure boundaries
- Integration with the existing Phase 2 training/evaluation stack

The FederatedClient is intentionally tested as an orchestration
layer. The tests do not replace Trainer, Evaluator, DataLoader, or
the Phase 3.1 parameter contract with duplicate implementations.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import SGD

from src.common.exceptions import FederatedLearningError
from src.data.dataset import FedMedDataset
from src.data.loader import create_dataloader
from src.fl.client import (
    FederatedClient,
    FederatedEvaluateResult,
    FederatedFitResult,
)
from src.fl.parameters import extract_parameters
from src.models.base_model import BaseModel
from src.training.evaluator import Evaluator
from src.training.metrics import Accuracy
from src.training.trainer import Trainer


# ============================================================
# Test model
# ============================================================


class ClientTestModel(BaseModel):
    """
    Small deterministic classification model used for client tests.
    """

    def build(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )


# ============================================================
# Fixtures / helpers
# ============================================================


@pytest.fixture
def model() -> ClientTestModel:
    """Create a fresh deterministic test model."""

    torch.manual_seed(42)

    return ClientTestModel(
        name="client_test_model",
        device="cpu",
    )


@pytest.fixture
def train_dataset() -> FedMedDataset:
    """Create a small local training dataset."""

    torch.manual_seed(10)

    samples = torch.randn(12, 4)

    targets = torch.tensor(
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        dtype=torch.long,
    )

    return FedMedDataset(
        samples=samples,
        targets=targets,
        name="client_train_dataset",
    )


@pytest.fixture
def eval_dataset() -> FedMedDataset:
    """Create a small local evaluation dataset."""

    torch.manual_seed(20)

    samples = torch.randn(8, 4)

    targets = torch.tensor(
        [0, 1, 0, 1, 0, 1, 0, 1],
        dtype=torch.long,
    )

    return FedMedDataset(
        samples=samples,
        targets=targets,
        name="client_eval_dataset",
    )


@pytest.fixture
def train_loader(
    train_dataset: FedMedDataset,
):
    """Create the local training DataLoader."""

    return create_dataloader(
        train_dataset,
        batch_size=4,
        shuffle=False,
    )


@pytest.fixture
def eval_loader(
    eval_dataset: FedMedDataset,
):
    """Create the local evaluation DataLoader."""

    return create_dataloader(
        eval_dataset,
        batch_size=4,
        shuffle=False,
    )


@pytest.fixture
def criterion() -> nn.Module:
    """Classification loss used by Trainer and Evaluator."""

    return nn.CrossEntropyLoss()


@pytest.fixture
def optimizer(model: ClientTestModel):
    """Create the local optimizer."""

    return SGD(
        model.parameters(),
        lr=0.01,
    )


@pytest.fixture
def trainer(
    model: ClientTestModel,
    criterion: nn.Module,
    optimizer,
):
    """Create the existing Phase 2 Trainer."""

    return Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epochs=1,
    )


@pytest.fixture
def evaluator(
    model: ClientTestModel,
    criterion: nn.Module,
):
    """Create the existing Phase 2 Evaluator."""

    return Evaluator(
        model=model,
        criterion=criterion,
        metrics=[Accuracy()],
    )


@pytest.fixture
def client(
    model: ClientTestModel,
    trainer: Trainer,
    evaluator: Evaluator,
    train_loader,
    eval_loader,
):
    """Create a fully configured FederatedClient."""

    return FederatedClient(
        client_id="client-1",
        model=model,
        trainer=trainer,
        evaluator=evaluator,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )


# ============================================================
# Construction
# ============================================================


def test_client_constructs_successfully(client: FederatedClient) -> None:
    """Verify a valid client can be constructed."""

    assert isinstance(client, FederatedClient)
    assert client.client_id == "client-1"
    assert client.has_evaluator is True


def test_client_exposes_parameter_contract(
    client: FederatedClient,
) -> None:
    """Verify the client exposes its immutable parameter contract."""

    contract = client.parameter_contract

    assert contract.count > 0
    assert len(contract.names) == contract.count
    assert len(contract.shapes) == contract.count


def test_client_rejects_non_string_client_id(
    model: ClientTestModel,
    trainer: Trainer,
    evaluator: Evaluator,
    train_loader,
) -> None:
    """Verify client_id must be a string."""

    with pytest.raises(FederatedLearningError):
        FederatedClient(
            client_id=123,  # type: ignore[arg-type]
            model=model,
            trainer=trainer,
            evaluator=evaluator,
            train_loader=train_loader,
        )


def test_client_rejects_empty_client_id(
    model: ClientTestModel,
    trainer: Trainer,
    evaluator: Evaluator,
    train_loader,
) -> None:
    """Verify blank client identifiers are rejected."""

    with pytest.raises(FederatedLearningError):
        FederatedClient(
            client_id="   ",
            model=model,
            trainer=trainer,
            evaluator=evaluator,
            train_loader=train_loader,
        )


def test_client_rejects_invalid_model(
    trainer: Trainer,
    evaluator: Evaluator,
    train_loader,
) -> None:
    """Verify arbitrary objects cannot be used as the model."""

    with pytest.raises(FederatedLearningError):
        FederatedClient(
            client_id="client-1",
            model=object(),  # type: ignore[arg-type]
            trainer=trainer,
            evaluator=evaluator,
            train_loader=train_loader,
        )


def test_client_rejects_invalid_trainer(
    model: ClientTestModel,
    evaluator: Evaluator,
    train_loader,
) -> None:
    """Verify Trainer is a required dependency."""

    with pytest.raises(FederatedLearningError):
        FederatedClient(
            client_id="client-1",
            model=model,
            trainer=object(),  # type: ignore[arg-type]
            evaluator=evaluator,
            train_loader=train_loader,
        )


def test_client_rejects_invalid_evaluator(
    model: ClientTestModel,
    trainer: Trainer,
    train_loader,
) -> None:
    """Verify Evaluator is a required dependency."""

    with pytest.raises(FederatedLearningError):
        FederatedClient(
            client_id="client-1",
            model=model,
            trainer=trainer,
            evaluator=object(),  # type: ignore[arg-type]
            train_loader=train_loader,
        )


def test_client_rejects_invalid_train_loader(
    model: ClientTestModel,
    trainer: Trainer,
    evaluator: Evaluator,
) -> None:
    """Verify a real DataLoader is required."""

    with pytest.raises(FederatedLearningError):
        FederatedClient(
            client_id="client-1",
            model=model,
            trainer=trainer,
            evaluator=evaluator,
            train_loader=object(),  # type: ignore[arg-type]
        )


def test_client_rejects_invalid_eval_loader(
    model: ClientTestModel,
    trainer: Trainer,
    evaluator: Evaluator,
    train_loader,
) -> None:
    """Verify invalid evaluation loaders are rejected."""

    with pytest.raises(FederatedLearningError):
        FederatedClient(
            client_id="client-1",
            model=model,
            trainer=trainer,
            evaluator=evaluator,
            train_loader=train_loader,
            eval_loader=object(),  # type: ignore[arg-type]
        )


def test_client_allows_missing_eval_loader(
    model: ClientTestModel,
    trainer: Trainer,
    evaluator: Evaluator,
    train_loader,
) -> None:
    """Verify evaluation is optional at client construction."""

    client = FederatedClient(
        client_id="client-1",
        model=model,
        trainer=trainer,
        evaluator=evaluator,
        train_loader=train_loader,
    )

    assert client.has_evaluator is False


def test_client_rejects_trainer_bound_to_different_model(
    model: ClientTestModel,
    criterion: nn.Module,
    evaluator: Evaluator,
    train_loader,
) -> None:
    """Verify Trainer and client must use the same model instance."""

    other_model = ClientTestModel(
        name="other_model",
        device="cpu",
    )

    trainer = Trainer(
        model=other_model,
        criterion=criterion,
        optimizer=SGD(other_model.parameters(), lr=0.01),
        epochs=1,
    )

    with pytest.raises(FederatedLearningError):
        FederatedClient(
            client_id="client-1",
            model=model,
            trainer=trainer,
            evaluator=evaluator,
            train_loader=train_loader,
        )


def test_client_rejects_evaluator_bound_to_different_model(
    model: ClientTestModel,
    criterion: nn.Module,
    trainer: Trainer,
    train_loader,
) -> None:
    """Verify Evaluator and client must use the same model instance."""

    other_model = ClientTestModel(
        name="other_model",
        device="cpu",
    )

    evaluator = Evaluator(
        model=other_model,
        criterion=criterion,
        metrics=[Accuracy()],
    )

    with pytest.raises(FederatedLearningError):
        FederatedClient(
            client_id="client-1",
            model=model,
            trainer=trainer,
            evaluator=evaluator,
            train_loader=train_loader,
        )


def test_client_rejects_empty_training_dataset(
    model: ClientTestModel,
    trainer: Trainer,
    evaluator: Evaluator,
):
    """Verify an empty local training dataset is rejected."""

    dataset = FedMedDataset(
        samples=torch.empty(0, 4),
        targets=torch.empty(0, dtype=torch.long),
        name="empty_train_dataset",
    )

    loader = create_dataloader(
        dataset,
        batch_size=4,
    )

    with pytest.raises(FederatedLearningError):
        FederatedClient(
            client_id="client-1",
            model=model,
            trainer=trainer,
            evaluator=evaluator,
            train_loader=loader,
        )


def test_client_rejects_empty_evaluation_dataset(
    model: ClientTestModel,
    trainer: Trainer,
    evaluator: Evaluator,
    train_loader,
):
    """Verify an empty evaluation dataset is rejected."""

    dataset = FedMedDataset(
        samples=torch.empty(0, 4),
        targets=torch.empty(0, dtype=torch.long),
        name="empty_eval_dataset",
    )

    loader = create_dataloader(
        dataset,
        batch_size=4,
    )

    with pytest.raises(FederatedLearningError):
        FederatedClient(
            client_id="client-1",
            model=model,
            trainer=trainer,
            evaluator=evaluator,
            train_loader=train_loader,
            eval_loader=loader,
        )


# ============================================================
# Parameters
# ============================================================


def test_get_parameters_returns_valid_payload(
    client: FederatedClient,
) -> None:
    """Verify get_parameters returns a valid NumPy payload."""

    parameters = client.get_parameters()

    assert isinstance(parameters, list)
    assert len(parameters) == client.parameter_contract.count

    assert all(
        isinstance(parameter, np.ndarray)
        for parameter in parameters
    )


def test_get_parameters_matches_model(
    client: FederatedClient,
) -> None:
    """Verify returned parameters represent the current model."""

    parameters = client.get_parameters()
    expected = extract_parameters(client._model)

    assert len(parameters) == len(expected)

    for actual, expected_array in zip(parameters, expected):
        np.testing.assert_array_equal(
            actual,
            expected_array,
        )


def test_get_parameters_is_defensive_copy(
    client: FederatedClient,
) -> None:
    """
    Verify modifying returned parameters does not modify model state.
    """

    parameters = client.get_parameters()
    original = client.get_parameters()

    parameters[0].flat[0] += 1000.0

    current = client.get_parameters()

    for expected, actual in zip(original, current):
        np.testing.assert_array_equal(
            expected,
            actual,
        )


def test_set_parameters_loads_global_parameters(
    client: FederatedClient,
) -> None:
    """Verify global parameters are loaded into the client model."""

    original = client.get_parameters()

    modified = [
        parameter.copy()
        for parameter in original
    ]

    modified[0] = modified[0] + 1.0

    client.set_parameters(modified)

    current = client.get_parameters()

    np.testing.assert_array_equal(
        current[0],
        modified[0],
    )


def test_set_parameters_does_not_retain_input_reference(
    client: FederatedClient,
) -> None:
    """
    Verify modifying the caller's parameter arrays after set_parameters()
    does not modify the client's model.
    """

    parameters = client.get_parameters()
    expected = [
        parameter.copy()
        for parameter in parameters
    ]

    client.set_parameters(parameters)

    parameters[0].fill(999.0)

    current = client.get_parameters()

    for expected_array, actual_array in zip(
        expected,
        current,
    ):
        np.testing.assert_array_equal(
            expected_array,
            actual_array,
        )


def test_set_parameters_rejects_count_mismatch(
    client: FederatedClient,
) -> None:
    """Verify invalid parameter count is rejected."""

    parameters = client.get_parameters()
    parameters.pop()

    with pytest.raises(FederatedLearningError):
        client.set_parameters(parameters)


def test_set_parameters_rejects_shape_mismatch(
    client: FederatedClient,
) -> None:
    """Verify invalid parameter shape is rejected."""

    parameters = client.get_parameters()

    parameters[0] = np.zeros(
        (999, 999),
        dtype=parameters[0].dtype,
    )

    with pytest.raises(FederatedLearningError):
        client.set_parameters(parameters)


def test_set_parameters_rejects_dtype_mismatch(
    client: FederatedClient,
) -> None:
    """Verify invalid parameter dtype is rejected."""

    parameters = client.get_parameters()

    parameters[0] = parameters[0].astype(np.float64)

    with pytest.raises(FederatedLearningError):
        client.set_parameters(parameters)


def test_set_parameters_rejects_nan(
    client: FederatedClient,
) -> None:
    """Verify NaN model parameters are rejected."""

    parameters = client.get_parameters()

    parameters[0].flat[0] = np.nan

    with pytest.raises(FederatedLearningError):
        client.set_parameters(parameters)


def test_set_parameters_rejects_positive_infinity(
    client: FederatedClient,
) -> None:
    """Verify positive infinity is rejected."""

    parameters = client.get_parameters()

    parameters[0].flat[0] = np.inf

    with pytest.raises(FederatedLearningError):
        client.set_parameters(parameters)


# ============================================================
# Fit
# ============================================================


def test_fit_returns_federated_fit_result(
    client: FederatedClient,
) -> None:
    """Verify fit returns the correct result type."""

    parameters = client.get_parameters()

    result = client.fit(parameters)

    assert isinstance(result, FederatedFitResult)


def test_fit_returns_updated_parameters(
    client: FederatedClient,
) -> None:
    """Verify fit returns a complete updated parameter payload."""

    parameters = client.get_parameters()

    result = client.fit(parameters)

    assert len(result.parameters) == client.parameter_contract.count

    assert all(
        isinstance(parameter, np.ndarray)
        for parameter in result.parameters
    )


def test_fit_reports_correct_sample_count(
    client: FederatedClient,
    train_dataset: FedMedDataset,
) -> None:
    """Verify training sample count comes from TrainingResult."""

    parameters = client.get_parameters()

    result = client.fit(parameters)

    assert result.num_examples == len(train_dataset)


def test_fit_reports_completed_epochs(
    client: FederatedClient,
) -> None:
    """Verify local epoch count is propagated."""

    parameters = client.get_parameters()

    result = client.fit(parameters)

    assert result.epochs_completed == 1


def test_fit_reports_processed_batches(
    client: FederatedClient,
) -> None:
    """Verify local batch count is propagated."""

    parameters = client.get_parameters()

    result = client.fit(parameters)

    assert result.batches_processed == 3


def test_fit_reports_final_loss(
    client: FederatedClient,
) -> None:
    """Verify final loss is propagated."""

    parameters = client.get_parameters()

    result = client.fit(parameters)

    assert isinstance(result.final_loss, float)
    assert np.isfinite(result.final_loss)
    assert result.final_loss >= 0.0


def test_fit_reports_training_metrics(
    client: FederatedClient,
) -> None:
    """Verify training metrics contain the expected values."""

    parameters = client.get_parameters()

    result = client.fit(parameters)

    assert "train_loss" in result.metrics
    assert "epochs_completed" in result.metrics
    assert "batches_processed" in result.metrics

    assert result.metrics["train_loss"] == pytest.approx(
        result.final_loss
    )

    assert result.metrics["epochs_completed"] == pytest.approx(
        result.epochs_completed
    )

    assert result.metrics["batches_processed"] == pytest.approx(
        result.batches_processed
    )


def test_fit_rejects_invalid_global_parameters(
    client: FederatedClient,
) -> None:
    """Verify fit validates global parameters before training."""

    parameters = client.get_parameters()
    parameters.pop()

    with pytest.raises(FederatedLearningError):
        client.fit(parameters)


def test_fit_does_not_mutate_input_parameters(
    client: FederatedClient,
) -> None:
    """Verify fit does not mutate the caller's parameter payload."""

    parameters = client.get_parameters()

    original = [
        parameter.copy()
        for parameter in parameters
    ]

    client.fit(parameters)

    for expected, actual in zip(
        original,
        parameters,
    ):
        np.testing.assert_array_equal(
            expected,
            actual,
        )


# ============================================================
# Evaluate
# ============================================================


def test_evaluate_returns_federated_evaluate_result(
    client: FederatedClient,
) -> None:
    """Verify evaluate returns the correct result type."""

    parameters = client.get_parameters()

    result = client.evaluate(parameters)

    assert isinstance(result, FederatedEvaluateResult)


def test_evaluate_reports_correct_sample_count(
    client: FederatedClient,
    eval_dataset: FedMedDataset,
) -> None:
    """Verify evaluation sample count is propagated correctly."""

    parameters = client.get_parameters()

    result = client.evaluate(parameters)

    assert result.num_examples == len(eval_dataset)


def test_evaluate_reports_finite_loss(
    client: FederatedClient,
) -> None:
    """Verify evaluation returns a finite non-negative loss."""

    parameters = client.get_parameters()

    result = client.evaluate(parameters)

    assert isinstance(result.loss, float)
    assert np.isfinite(result.loss)
    assert result.loss >= 0.0


def test_evaluate_reports_accuracy(
    client: FederatedClient,
) -> None:
    """Verify evaluator metrics are propagated."""

    parameters = client.get_parameters()

    result = client.evaluate(parameters)

    assert "accuracy" in result.metrics

    assert 0.0 <= result.metrics["accuracy"] <= 1.0


def test_evaluate_rejects_invalid_global_parameters(
    client: FederatedClient,
) -> None:
    """Verify evaluate validates global parameters before evaluation."""

    parameters = client.get_parameters()
    parameters.pop()

    with pytest.raises(FederatedLearningError):
        client.evaluate(parameters)


def test_evaluate_requires_eval_loader(
    model: ClientTestModel,
    trainer: Trainer,
    evaluator: Evaluator,
    train_loader,
) -> None:
    """Verify evaluation fails clearly when no eval loader exists."""

    client = FederatedClient(
        client_id="client-without-eval",
        model=model,
        trainer=trainer,
        evaluator=evaluator,
        train_loader=train_loader,
    )

    parameters = client.get_parameters()

    with pytest.raises(FederatedLearningError):
        client.evaluate(parameters)


# ============================================================
# Fit / evaluate state behavior
# ============================================================


def test_fit_then_evaluate_uses_updated_local_model(
    client: FederatedClient,
) -> None:
    """
    Verify the client can train and then evaluate the resulting
    local model without reconstructing the client.
    """

    parameters = client.get_parameters()

    fit_result = client.fit(parameters)

    evaluation_result = client.evaluate(
        fit_result.parameters
    )

    assert evaluation_result.num_examples > 0
    assert np.isfinite(evaluation_result.loss)
    assert 0.0 <= evaluation_result.metrics["accuracy"] <= 1.0


def test_client_can_load_new_global_parameters_and_evaluate(
    client: FederatedClient,
) -> None:
    """
    Verify a later global model can replace the client's local state
    before evaluation.
    """

    original = client.get_parameters()

    updated = [
        parameter.copy()
        for parameter in original
    ]

    updated[0] = updated[0] + 0.5

    client.set_parameters(updated)

    evaluation_result = client.evaluate(updated)

    assert evaluation_result.num_examples > 0
    assert np.isfinite(evaluation_result.loss)


# ============================================================
# Result isolation
# ============================================================


def test_fit_result_parameters_are_defensive(
    client: FederatedClient,
) -> None:
    """
    Verify mutating the returned FitResult parameter payload does not
    mutate the client's model.
    """

    parameters = client.get_parameters()

    result = client.fit(parameters)

    current_before = client.get_parameters()

    result.parameters[0].flat[0] += 1000.0

    current_after = client.get_parameters()

    for before, after in zip(
        current_before,
        current_after,
    ):
        np.testing.assert_array_equal(
            before,
            after,
        )


def test_evaluate_metrics_are_independent_mapping(
    client: FederatedClient,
) -> None:
    """Verify evaluation returns a fresh metric mapping."""

    parameters = client.get_parameters()

    first = client.evaluate(parameters)
    second = client.evaluate(parameters)

    assert first.metrics == second.metrics
    assert first.metrics is not second.metrics