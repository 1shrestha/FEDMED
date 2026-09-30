"""
Tests for the FedMed local evaluation engine (src/training/evaluator.py).

Uses a tiny synthetic classification model/dataset. No medical data.
"""

import pytest
import torch
from torch import nn

from src.common.exceptions import TrainingError
from src.data.dataset import FedMedDataset
from src.data.loader import create_dataloader
from src.models.base_model import BaseModel
from src.training.evaluator import (
    EvaluationResult,
    Evaluator,
    _infer_batch_size,
    _move_to_device,
)
from src.training.metrics import Accuracy


class TinyClassifier(BaseModel):
    """Minimal synthetic classification model used only for testing."""

    def build(self) -> nn.Module:
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def _make_synthetic_dataset(size: int = 16, name: str = "eval_synthetic") -> FedMedDataset:
    torch.manual_seed(1)
    samples = [torch.randn(4) for _ in range(size)]
    targets = [i % 2 for i in range(size)]
    return FedMedDataset(samples=samples, targets=targets, name=name)


@pytest.fixture
def model() -> TinyClassifier:
    return TinyClassifier(name="tiny_eval_classifier", device="cpu")


@pytest.fixture
def criterion():
    return nn.CrossEntropyLoss()


@pytest.fixture
def dataloader():
    dataset = _make_synthetic_dataset(size=16)
    return create_dataloader(dataset, batch_size=4, shuffle=False)


# ============================================================
# Construction validation
# ============================================================


def test_valid_evaluator_construction(model, criterion):
    evaluator = Evaluator(model=model, criterion=criterion)
    assert isinstance(evaluator, Evaluator)


def test_evaluator_with_metrics(model, criterion):
    evaluator = Evaluator(model=model, criterion=criterion, metrics=[Accuracy()])
    assert isinstance(evaluator, Evaluator)


def test_invalid_model_rejected(criterion):
    with pytest.raises(TrainingError):
        Evaluator(model=object(), criterion=criterion)


def test_invalid_criterion_rejected(model):
    with pytest.raises(TrainingError):
        Evaluator(model=model, criterion="not_callable")


def test_invalid_metrics_rejected(model, criterion):
    with pytest.raises(TrainingError):
        Evaluator(model=model, criterion=criterion, metrics=["not_a_metric"])


def test_invalid_dataloader_rejected(model, criterion):
    evaluator = Evaluator(model=model, criterion=criterion)

    with pytest.raises(TrainingError):
        evaluator.evaluate([1, 2, 3])


def test_duplicate_metric_names_rejected(model, criterion):
    """Two metrics sharing a name must be rejected at construction, not
    silently overwrite each other in the results dict."""

    with pytest.raises(TrainingError):
        Evaluator(model=model, criterion=criterion, metrics=[Accuracy(), Accuracy()])


# ============================================================
# Evaluation behavior
# ============================================================


def test_valid_evaluation(model, criterion, dataloader):
    evaluator = Evaluator(model=model, criterion=criterion)

    result = evaluator.evaluate(dataloader)

    assert isinstance(result, EvaluationResult)


def test_evaluation_result_structure(model, criterion, dataloader):
    evaluator = Evaluator(model=model, criterion=criterion, metrics=[Accuracy()])

    result = evaluator.evaluate(dataloader)

    assert hasattr(result, "samples_evaluated")
    assert hasattr(result, "batches_evaluated")
    assert hasattr(result, "loss")
    assert hasattr(result, "metrics")
    assert isinstance(result.metrics, dict)


def test_sample_count(model, criterion, dataloader):
    evaluator = Evaluator(model=model, criterion=criterion)

    result = evaluator.evaluate(dataloader)

    assert result.samples_evaluated == 16


def test_batch_count(model, criterion, dataloader):
    evaluator = Evaluator(model=model, criterion=criterion)

    result = evaluator.evaluate(dataloader)

    # 16 samples / batch_size 4 = 4 batches
    assert result.batches_evaluated == 4


def test_evaluation_loss(model, criterion, dataloader):
    evaluator = Evaluator(model=model, criterion=criterion)

    result = evaluator.evaluate(dataloader)

    assert result.loss >= 0.0
    assert isinstance(result.loss, float)


def test_metrics_computed(model, criterion, dataloader):
    evaluator = Evaluator(model=model, criterion=criterion, metrics=[Accuracy()])

    result = evaluator.evaluate(dataloader)

    assert "accuracy" in result.metrics
    assert 0.0 <= result.metrics["accuracy"] <= 1.0


def test_parameters_remain_unchanged(model, criterion, dataloader):
    """Verify evaluation never updates model parameters."""

    original_params = [p.clone() for p in model.network.parameters()]

    evaluator = Evaluator(model=model, criterion=criterion)
    evaluator.evaluate(dataloader)

    updated_params = list(model.network.parameters())

    assert all(
        torch.equal(original, updated)
        for original, updated in zip(original_params, updated_params)
    )


def test_gradients_are_not_accumulated(model, criterion, dataloader):
    """Verify no gradients are populated on model parameters after evaluation."""

    evaluator = Evaluator(model=model, criterion=criterion)
    evaluator.evaluate(dataloader)

    for param in model.network.parameters():
        assert param.grad is None


def test_optimizer_state_not_touched_by_evaluation(model, criterion, dataloader):
    """Verify an optimizer sitting alongside the model is never stepped or
    populated by evaluation."""

    optimizer = torch.optim.SGD(model.network.parameters(), lr=0.01)
    params_before = [p.clone() for p in model.network.parameters()]

    evaluator = Evaluator(model=model, criterion=criterion)
    evaluator.evaluate(dataloader)

    assert len(optimizer.state) == 0
    assert all(
        torch.equal(before, after)
        for before, after in zip(params_before, model.network.parameters())
    )


def test_training_mode_is_restored(model, criterion, dataloader):
    """Verify a model that was in training mode returns to training mode."""

    model.train_mode()
    assert model.network.training is True

    evaluator = Evaluator(model=model, criterion=criterion)
    evaluator.evaluate(dataloader)

    assert model.network.training is True


def test_eval_mode_remains_eval(model, criterion, dataloader):
    """Verify a model that was already in eval mode remains in eval mode
    (not incorrectly flipped to train mode)."""

    model.eval_mode()
    assert model.network.training is False

    evaluator = Evaluator(model=model, criterion=criterion)
    evaluator.evaluate(dataloader)

    assert model.network.training is False


def test_evaluation_failure_becomes_training_error(model, dataloader):
    """Verify a criterion that raises mid-evaluation surfaces as
    TrainingError."""

    def broken_criterion(outputs, targets):
        raise RuntimeError("simulated criterion failure")

    evaluator = Evaluator(model=model, criterion=broken_criterion)

    with pytest.raises(TrainingError):
        evaluator.evaluate(dataloader)


def test_evaluation_exception_is_chained(model, dataloader):
    """Verify the original exception is chained via __cause__."""

    def broken_criterion(outputs, targets):
        raise RuntimeError("simulated criterion failure")

    evaluator = Evaluator(model=model, criterion=broken_criterion)

    with pytest.raises(TrainingError) as excinfo:
        evaluator.evaluate(dataloader)

    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_mode_restored_even_after_failure(model, dataloader):
    """Verify model mode is restored even when evaluation raises."""

    model.train_mode()

    def broken_criterion(outputs, targets):
        raise RuntimeError("simulated criterion failure")

    evaluator = Evaluator(model=model, criterion=broken_criterion)

    with pytest.raises(TrainingError):
        evaluator.evaluate(dataloader)

    assert model.network.training is True


# ============================================================
# Loss validation
# ============================================================


def test_non_scalar_evaluation_loss_rejected(model, dataloader):
    def bad_criterion(outputs, targets):
        return outputs.sum(dim=0)

    evaluator = Evaluator(model=model, criterion=bad_criterion)

    with pytest.raises(TrainingError):
        evaluator.evaluate(dataloader)


def test_non_finite_evaluation_loss_rejected(model, dataloader):
    def nan_criterion(outputs, targets):
        return outputs.sum() * float("nan")

    evaluator = Evaluator(model=model, criterion=nan_criterion)

    with pytest.raises(TrainingError):
        evaluator.evaluate(dataloader)


# ============================================================
# Sample-weighted loss / empty DataLoader
# ============================================================


def test_uneven_batch_sample_weighted_evaluation_loss(model):
    """
    Deterministic proof that EvaluationResult.loss is sample-weighted,
    not a naive average of per-batch means. 5 samples, batch_size=3 ->
    batches of size 3 and 2, with a fixed-value criterion.
    """

    dataset = _make_synthetic_dataset(size=5, name="eval_uneven_loss_dataset")
    loader = create_dataloader(dataset, batch_size=3, shuffle=False, drop_last=False)

    loss_by_batch_size = {3: 0.2, 2: 1.0}

    def fixed_value_criterion(outputs, targets):
        batch_size = outputs.shape[0]
        return outputs.sum() * 0.0 + torch.tensor(
            loss_by_batch_size[batch_size], dtype=outputs.dtype
        )

    evaluator = Evaluator(model=model, criterion=fixed_value_criterion)
    result = evaluator.evaluate(loader)

    expected_sample_weighted_loss = (0.2 * 3 + 1.0 * 2) / 5
    naive_unweighted_average = (0.2 + 1.0) / 2

    assert result.loss == pytest.approx(expected_sample_weighted_loss)
    assert result.loss != pytest.approx(naive_unweighted_average)


def test_zero_batch_evaluation_raises(model, criterion):
    """batch_size larger than the dataset with drop_last=True yields zero
    batches; evaluation must reject this rather than reporting fake results."""

    dataset = _make_synthetic_dataset(size=5, name="eval_zero_batch_dataset")
    loader = create_dataloader(dataset, batch_size=10, drop_last=True)

    evaluator = Evaluator(model=model, criterion=criterion)

    with pytest.raises(TrainingError):
        evaluator.evaluate(loader)


# ============================================================
# Metric state isolation (mandatory: Section 46)
# ============================================================


def test_metric_state_does_not_leak_between_evaluations(model, criterion):
    """
    Running the same Evaluator/metric instance across two evaluations
    must not let the first evaluation's accumulated state leak into
    the second. Verified by comparing a reused-instance evaluation of
    dataset B against a completely independent fresh evaluation of the
    same dataset B: leaking state would make them diverge.
    """

    dataset_a = _make_synthetic_dataset(size=8, name="isolation_dataset_a")
    loader_a = create_dataloader(dataset_a, batch_size=4, shuffle=False)

    dataset_b = _make_synthetic_dataset(size=4, name="isolation_dataset_b")
    loader_b = create_dataloader(dataset_b, batch_size=4, shuffle=False)

    shared_metric = Accuracy()
    evaluator = Evaluator(model=model, criterion=criterion, metrics=[shared_metric])

    evaluator.evaluate(loader_a)
    result_b_from_shared_evaluator = evaluator.evaluate(loader_b)

    fresh_evaluator = Evaluator(model=model, criterion=criterion, metrics=[Accuracy()])
    result_b_from_fresh_evaluator = fresh_evaluator.evaluate(loader_b)

    assert result_b_from_shared_evaluator.samples_evaluated == 4
    assert result_b_from_fresh_evaluator.samples_evaluated == 4
    assert result_b_from_shared_evaluator.metrics["accuracy"] == pytest.approx(
        result_b_from_fresh_evaluator.metrics["accuracy"]
    )


# ============================================================
# Internal helpers (direct unit tests)
# ============================================================


def test_move_to_device_handles_nested_dict_list_tuple():
    """_move_to_device must recurse through dict/list/tuple, preserving
    structure and non-tensor leaves."""

    device = torch.device("cpu")

    batch = {
        "image": torch.randn(2, 3),
        "meta": [torch.randn(2), torch.randn(2)],
        "extra": (torch.randn(2), "non_tensor_value"),
    }

    moved = _move_to_device(batch, device)

    assert isinstance(moved, dict)
    assert moved["image"].device == device
    assert isinstance(moved["meta"], list)
    assert all(t.device == device for t in moved["meta"])
    assert isinstance(moved["extra"], tuple)
    assert moved["extra"][1] == "non_tensor_value"


def test_infer_batch_size_rejects_scalar_tensor():
    """A 0-dimensional Tensor has no batch dimension and must raise
    TrainingError rather than crashing on shape[0]."""

    scalar_tensor = torch.tensor(5.0)

    with pytest.raises(TrainingError):
        _infer_batch_size(scalar_tensor, scalar_tensor)


def test_infer_batch_size_from_dict_shaped_samples():
    """
    A mapping-shaped samples value must infer batch size from its
    tensor values, not from len(mapping), because len(mapping)
    represents the number of fields rather than the number of samples.
    """

    dict_batch = {
        "image": torch.randn(6, 3, 8, 8),
        "meta": torch.randn(6, 4),
    }

    targets = torch.randint(0, 2, (6,))

    assert _infer_batch_size(dict_batch, targets) == 6


def test_infer_batch_size_from_nested_mapping():
    """
    Batch-size inference must recurse through nested mappings used
    by structured or multimodal batches.
    """

    nested_batch = {
        "image": {
            "left": torch.randn(5, 3, 8, 8),
            "right": torch.randn(5, 3, 8, 8),
        },
        "metadata": {
            "clinical": torch.randn(5, 4),
        },
    }

    targets = torch.randint(0, 2, (5,))

    assert _infer_batch_size(nested_batch, targets) == 5


def test_infer_batch_size_falls_back_to_targets():
    """
    If samples cannot provide a batch size, inference must fall back
    to targets rather than using the number of mapping fields.
    """

    samples = {
        "metadata": "not_a_batch_dimension",
    }

    targets = torch.randint(0, 2, (7,))

    assert _infer_batch_size(samples, targets) == 7
    

def test_move_to_device_handles_mapping_subclass():
    """_move_to_device must support Mapping implementations, not only dict."""

    from collections import OrderedDict

    device = torch.device("cpu")

    batch = OrderedDict(
        [
            ("image", torch.randn(3, 4)),
            ("meta", torch.randn(3, 2)),
        ]
    )

    moved = _move_to_device(batch, device)

    assert isinstance(moved, dict)
    assert moved["image"].device == device
    assert moved["meta"].device == device
# ============================================================
# Mandatory end-to-end integration test
# ============================================================


def test_full_pipeline_dataset_to_evaluation(criterion):
    """
    Mandatory end-to-end test exercising the full Phase 2 chain:

        FedMedDataset -> partition_dataset() -> PartitionView ->
        create_dataloader() -> BaseModel -> Trainer -> trained model ->
        Evaluator -> EvaluationResult

    This verifies the components work together, not merely in
    isolated unit tests.
    """

    from src.common.config import TrainingConfig
    from src.data.partitioner import partition_dataset
    from src.training.trainer import Trainer

    dataset = _make_synthetic_dataset(size=40, name="full_pipeline_dataset")
    partitions = partition_dataset(dataset, num_clients=2, seed=3)

    client_partition = partitions["client_0"]

    train_loader = create_dataloader(
        client_partition.dataset, batch_size=5, shuffle=True
    )
    eval_loader = create_dataloader(
        client_partition.dataset, batch_size=5, shuffle=False
    )

    model = TinyClassifier(name="pipeline_model", device="cpu")

    config = TrainingConfig(
        local_epochs=2,
        batch_size=5,
        learning_rate=0.01,
        optimizer="adam",
        seed=42,
    )

    trainer = Trainer(model=model, criterion=criterion, config=config)
    training_result = trainer.train(train_loader)

    assert training_result.epochs_completed == 2
    assert training_result.samples_processed == 2 * len(client_partition.dataset)

    evaluator = Evaluator(model=model, criterion=criterion, metrics=[Accuracy()])
    evaluation_result = evaluator.evaluate(eval_loader)

    assert isinstance(evaluation_result, EvaluationResult)
    assert evaluation_result.samples_evaluated == len(client_partition.dataset)
    assert evaluation_result.loss >= 0.0
    assert 0.0 <= evaluation_result.metrics["accuracy"] <= 1.0