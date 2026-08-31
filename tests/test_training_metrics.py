"""
Tests for the FedMed metric foundation (src/training/metrics.py).

Covers:

- Metric abstract interface (via Accuracy)
- Single-batch computation
- Multi-batch accumulation
- Reset behavior
- Invalid input handling
- Correct numerical results
"""

import pytest
import torch

from src.common.exceptions import TrainingError
from src.training.metrics import Accuracy, Metric


def test_accuracy_is_a_metric() -> None:
    """Verify Accuracy implements the Metric interface."""

    metric = Accuracy()

    assert isinstance(metric, Metric)
    assert metric.name == "accuracy"


def test_accuracy_single_batch_all_correct() -> None:
    """Verify accuracy computes 1.0 when all predictions are correct."""

    metric = Accuracy()

    outputs = torch.tensor(
        [[10.0, 0.0], [0.0, 10.0], [10.0, 0.0]]
    )  # predicts class 0, 1, 0
    targets = torch.tensor([0, 1, 0])

    metric.update(outputs, targets)

    assert metric.compute() == pytest.approx(1.0)


def test_accuracy_single_batch_all_wrong() -> None:
    """Verify accuracy computes 0.0 when all predictions are wrong."""

    metric = Accuracy()

    outputs = torch.tensor([[10.0, 0.0], [0.0, 10.0]])  # predicts 0, 1
    targets = torch.tensor([1, 0])  # actually 1, 0 -> both wrong

    metric.update(outputs, targets)

    assert metric.compute() == pytest.approx(0.0)


def test_accuracy_single_batch_partial() -> None:
    """Verify a known partially-correct batch computes the exact fraction."""

    metric = Accuracy()

    outputs = torch.tensor(
        [[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]]
    )  # predicts 0, 1, 0, 1
    targets = torch.tensor([0, 1, 1, 1])  # 3 of 4 correct

    metric.update(outputs, targets)

    assert metric.compute() == pytest.approx(0.75)


def test_accuracy_multi_batch_accumulation() -> None:
    """
    Verify accuracy accumulates correctly across multiple update()
    calls rather than only reflecting the most recent batch.
    """

    metric = Accuracy()

    # Batch 1: 2/2 correct
    metric.update(
        torch.tensor([[10.0, 0.0], [0.0, 10.0]]),
        torch.tensor([0, 1]),
    )

    # Batch 2: 0/2 correct
    metric.update(
        torch.tensor([[10.0, 0.0], [0.0, 10.0]]),
        torch.tensor([1, 0]),
    )

    # Overall: 2/4 = 0.5
    assert metric.compute() == pytest.approx(0.5)


def test_accuracy_reset_clears_state() -> None:
    """Verify reset() returns the metric to its initial condition."""

    metric = Accuracy()

    metric.update(
        torch.tensor([[10.0, 0.0], [0.0, 10.0]]),
        torch.tensor([0, 1]),
    )
    assert metric.compute() == pytest.approx(1.0)

    metric.reset()

    with pytest.raises(TrainingError):
        metric.compute()

    # metric is usable again after reset
    metric.update(
        torch.tensor([[10.0, 0.0]]),
        torch.tensor([1]),
    )
    assert metric.compute() == pytest.approx(0.0)


def test_accuracy_compute_before_update_raises() -> None:
    """Verify compute() before any update() raises TrainingError."""

    metric = Accuracy()

    with pytest.raises(TrainingError):
        metric.compute()


def test_accuracy_rejects_non_tensor_outputs() -> None:
    """Verify non-tensor outputs raise TrainingError."""

    metric = Accuracy()

    with pytest.raises(TrainingError):
        metric.update([[1.0, 0.0]], torch.tensor([0]))


def test_accuracy_rejects_non_tensor_targets() -> None:
    """Verify non-tensor targets raise TrainingError."""

    metric = Accuracy()

    with pytest.raises(TrainingError):
        metric.update(torch.tensor([[1.0, 0.0]]), [0])


def test_accuracy_rejects_wrong_output_dims() -> None:
    """Verify 1-dimensional outputs (missing class dimension) raise
    TrainingError."""

    metric = Accuracy()

    with pytest.raises(TrainingError):
        metric.update(torch.tensor([1.0, 0.0, 1.0]), torch.tensor([0, 1, 0]))


def test_accuracy_rejects_batch_size_mismatch() -> None:
    """Verify mismatched outputs/targets batch sizes raise TrainingError."""

    metric = Accuracy()

    with pytest.raises(TrainingError):
        metric.update(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0]),
        )


def test_accuracy_rejects_wrong_target_dims() -> None:
    """Verify targets with ndim != 1 (e.g. shape (batch, 1)) raise
    TrainingError instead of silently broadcasting into a nonsensical
    comparison."""

    metric = Accuracy()

    with pytest.raises(TrainingError):
        metric.update(
            torch.tensor([[10.0, 0.0], [0.0, 10.0]]),
            torch.tensor([[0], [1]]),
        )


def test_accuracy_rejects_non_integer_targets() -> None:
    """Verify floating-point targets are rejected."""

    metric = Accuracy()

    outputs = torch.tensor(
        [[10.0, 0.0], [0.0, 10.0]]
    )
    targets = torch.tensor([0.0, 1.0])

    with pytest.raises(TrainingError):
        metric.update(outputs, targets)


def test_accuracy_rejects_target_class_out_of_range() -> None:
    """
    Verify target class indices outside the model's class range
    are rejected.
    """

    metric = Accuracy()

    outputs = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ]
    )

    # Valid classes are 0, 1, and 2.
    targets = torch.tensor([0, 3])

    with pytest.raises(TrainingError):
        metric.update(outputs, targets)


def test_accuracy_rejects_negative_target_class() -> None:
    """Verify negative class indices are rejected."""

    metric = Accuracy()

    outputs = torch.tensor(
        [
            [10.0, 0.0],
            [0.0, 10.0],
        ]
    )

    targets = torch.tensor([0, -1])

    with pytest.raises(TrainingError):
        metric.update(outputs, targets)



def test_accuracy_rejects_zero_class_outputs() -> None:
    """Verify outputs with zero classes (shape (batch, 0)) raise a clear
    TrainingError instead of failing inside argmax()."""

    metric = Accuracy()

    with pytest.raises(TrainingError):
        metric.update(torch.empty(3, 0), torch.tensor([0, 1, 0]))


def test_accuracy_reset_after_multiple_updates() -> None:
    """Verify reset() clears state accumulated across several updates."""

    metric = Accuracy()

    metric.update(torch.tensor([[10.0, 0.0]]), torch.tensor([0]))
    metric.update(torch.tensor([[0.0, 10.0]]), torch.tensor([0]))  # wrong

    assert metric.compute() == pytest.approx(0.5)

    metric.reset()

    with pytest.raises(TrainingError):
        metric.compute()


def test_accuracy_repeated_independent_usage() -> None:
    """Verify a metric instance can be reused for multiple independent
    evaluation passes via reset(), each producing correct isolated results."""

    metric = Accuracy()

    metric.update(torch.tensor([[10.0, 0.0]]), torch.tensor([0]))
    first_result = metric.compute()

    metric.reset()

    metric.update(torch.tensor([[0.0, 10.0]]), torch.tensor([1]))
    second_result = metric.compute()

    assert first_result == pytest.approx(1.0)
    assert second_result == pytest.approx(1.0)


def test_accuracy_result_always_in_valid_range() -> None:
    """Verify accuracy remains within [0.0, 1.0] for arbitrary random
    outputs/targets."""

    torch.manual_seed(7)
    metric = Accuracy()

    outputs = torch.randn(50, 4)
    targets = torch.randint(0, 4, (50,))

    metric.update(outputs, targets)
    result = metric.compute()

    assert 0.0 <= result <= 1.0