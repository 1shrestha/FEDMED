"""
Metric foundation for FedMed local training/evaluation.

This module defines Metric, a small generic interface for
accumulating a numerical measure across batches, plus one concrete
implementation (Accuracy) to demonstrate the interface with a real,
testable metric.

This module intentionally does NOT contain:

- A medical metrics library
- Flower-specific or federated-round-aware logic
- Hospital/client identity
- A plugin/registry framework (metrics are constructed directly by
  the caller and passed to Evaluator, mirroring how Trainer receives
  an already-constructed optimizer)
- Task-specific assumptions baked into the interface itself (Metric
  works for classification, regression, or segmentation outputs;
  only the concrete Accuracy implementation assumes a classification
  shape, and only because Accuracy specifically models that concept)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from src.common.exceptions import TrainingError


class Metric(ABC):
    """
    Abstract interface for an accumulating evaluation metric.

    Lifecycle:
        metric.reset()
        for each batch:
            metric.update(outputs, targets)
        result = metric.compute()

    Concrete metrics accumulate internal state across multiple
    update() calls (e.g. running correct/total counts) so that a
    single metric instance can summarize an entire epoch or
    evaluation pass, not just one batch.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this metric, used as a result key."""
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Incorporate one batch of model outputs and targets into the
        running metric state.

        Args:
            outputs: Model outputs for the batch.
            targets: Corresponding targets for the batch.

        Raises:
            TrainingError: If outputs/targets are invalid for this
                metric (see concrete implementation for specifics).
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> float:
        """
        Compute the current metric value from accumulated state.

        Returns:
            The metric's current value.

        Raises:
            TrainingError: If compute() is called before any update().
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Clear all accumulated state, returning the metric to its
        initial condition.
        """
        raise NotImplementedError


class Accuracy(Metric):
    """
    Classification accuracy: fraction of samples whose predicted
    class (argmax over the last output dimension) matches the target
    class index.

    Expects:
        outputs:
            Tensor of shape (batch_size, num_classes), with
            num_classes >= 1. Values may be raw logits or
            probabilities; softmax is not required because argmax
            determines the predicted class.

        targets:
            Tensor of shape (batch_size,) containing integer class
            indices in the range [0, num_classes - 1].

    This is one concrete, opt-in metric; it does not make Evaluator
    or Trainer assume classification. Callers working on regression
    or segmentation tasks simply do not pass an Accuracy instance.
    """

    _VALID_TARGET_DTYPES = (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    )

    def __init__(self) -> None:
        self._correct = 0
        self._total = 0

    @property
    def name(self) -> str:
        """Return the stable result key for this metric."""
        return "accuracy"

    def update(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Incorporate one batch into the running accuracy state.

        Validation is intentionally performed here, at the concrete
        metric boundary, so malformed classification inputs fail with
        a FedMed TrainingError instead of producing misleading
        results or low-level PyTorch errors.

        Args:
            outputs:
                Tensor of shape (batch_size, num_classes), where
                num_classes >= 1.

            targets:
                Tensor of shape (batch_size,) containing integer class
                indices.

        Raises:
            TrainingError:
                If outputs/targets are not tensors, have incompatible
                shapes, outputs is not 2-dimensional, outputs has zero
                classes, targets is not 1-dimensional, targets have a
                non-integer dtype, outputs and targets are on different
                devices, or target indices are outside the valid class
                range.
        """

        # ------------------------------------------------------------
        # Type validation
        # ------------------------------------------------------------

        if not isinstance(outputs, torch.Tensor) or not isinstance(
            targets,
            torch.Tensor,
        ):
            raise TrainingError(
                "Accuracy.update() requires torch.Tensor outputs and "
                f"targets, got {type(outputs).__name__} and "
                f"{type(targets).__name__}."
            )

        # ------------------------------------------------------------
        # Output shape validation
        # ------------------------------------------------------------

        if outputs.dim() != 2:
            raise TrainingError(
                "Accuracy.update() expects outputs of shape "
                "(batch_size, num_classes), got shape "
                f"{tuple(outputs.shape)}."
            )

        if outputs.shape[1] == 0:
            raise TrainingError(
                "Accuracy.update() requires outputs to have at least "
                f"one class, got shape {tuple(outputs.shape)}."
            )

        # ------------------------------------------------------------
        # Target shape validation
        # ------------------------------------------------------------

        if targets.dim() != 1:
            raise TrainingError(
                "Accuracy.update() expects targets of shape "
                f"(batch_size,), got shape {tuple(targets.shape)}."
            )

        if outputs.shape[0] != targets.shape[0]:
            raise TrainingError(
                "Accuracy.update() batch size mismatch: outputs has "
                f"{outputs.shape[0]} samples, targets has "
                f"{targets.shape[0]}."
            )

        # ------------------------------------------------------------
        # Target dtype validation
        # ------------------------------------------------------------

        if targets.dtype not in self._VALID_TARGET_DTYPES:
            raise TrainingError(
                "Accuracy.update() expects integer class-index "
                f"targets, got dtype {targets.dtype}."
            )

        # ------------------------------------------------------------
        # Device validation
        # ------------------------------------------------------------

        if outputs.device != targets.device:
            raise TrainingError(
                "Accuracy.update() requires outputs and targets to "
                f"be on the same device, got outputs on "
                f"{outputs.device} and targets on "
                f"{targets.device}."
            )

        # ------------------------------------------------------------
        # Target class-range validation
        # ------------------------------------------------------------

        if targets.numel() > 0:
            invalid_targets = (targets < 0) | (
                targets >= outputs.shape[1]
            )

            if torch.any(invalid_targets):
                raise TrainingError(
                    "Accuracy.update() found target class indices "
                    "outside the valid range "
                    f"[0, {outputs.shape[1] - 1}]."
                )

        # ------------------------------------------------------------
        # Metric accumulation
        # ------------------------------------------------------------

        with torch.no_grad():
            predictions = outputs.argmax(dim=-1)

            self._correct += int(
                (predictions == targets).sum().item()
            )

            self._total += int(targets.shape[0])

    def compute(self) -> float:
        """
        Compute accumulated classification accuracy.

        Returns:
            Fraction of correctly predicted samples across all
            update() calls since the last reset(), in [0.0, 1.0].

        Raises:
            TrainingError:
                If compute() is called before any update().
        """

        if self._total == 0:
            raise TrainingError(
                "Accuracy.compute() called before any update(); "
                "accuracy is undefined with zero samples."
            )

        return self._correct / self._total

    def reset(self) -> None:
        """
        Reset accumulated correct and total sample counts.
        """

        self._correct = 0
        self._total = 0