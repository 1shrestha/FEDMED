"""
Local evaluation engine for FedMed.

This module defines Evaluator, which runs a no-gradient forward pass
over an already-constructed BaseModel and DataLoader, computing loss
and optional metrics without updating any model parameter.

Evaluator is PHASE 2 = LOCAL LEARNING only, same as trainer.py. It
has no knowledge of federated rounds, clients, hospitals,
aggregation, or Flower.

This module intentionally does NOT:

- update model parameters, call optimizer.step(), or call backward()
- create datasets, partitions, or DataLoaders
- create or register models
- perform federated aggregation or communicate with Flower

Loss aggregation contract:

    Mirrors trainer.py exactly: the criterion is expected to return
    the mean loss over the current batch, and EvaluationResult.loss
    is the SAMPLE-WEIGHTED mean across all batches (not a simple
    average of per-batch means), since batches may have unequal
    sizes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from src.common.exceptions import TrainingError
from src.common.logging import get_logger
from src.models.base_model import BaseModel
from src.training.metrics import Metric

logger = get_logger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    """
    Structured outcome of an Evaluator.evaluate() call.

    Deliberately contains only local-evaluation facts — no federated
    concepts (client_id, round number, aggregation weight, etc.).

    Attributes:
        samples_evaluated: Total number of samples seen.
        batches_evaluated: Total number of batches processed.
        loss: Sample-weighted mean loss across all evaluated samples
            (see module docstring "Loss aggregation contract").
        metrics: Mapping of metric name -> computed value, for every
            Metric supplied at Evaluator construction. Empty dict if
            none were supplied.
    """

    samples_evaluated: int
    batches_evaluated: int
    loss: float
    metrics: dict[str, float] = field(default_factory=dict)


def _move_to_device(value: Any, device: torch.device) -> Any:
    """
    Recursively move tensor-containing batch structures to a device.

    Supports the standard PyTorch batch containers produced by
    default_collate: Tensor, dict, list, tuple. Structure is
    preserved; nothing is mutated in place. Mirrors trainer.py's
    helper of the same behavior — intentionally small and duplicated
    rather than introducing a second shared device-management module
    for a handful of lines.
    """

    if isinstance(value, torch.Tensor):
        return value.to(device)

    if isinstance(value, Mapping):
        return {
            key: _move_to_device(item, device)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]

    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)

    return value


def _infer_batch_size(samples: Any, targets: Any) -> int:
    """
    Determine how many samples are in a batch, for statistics only.

    Samples are inspected first, then targets.

    Tensor values use dimension 0 as the batch dimension.

    Mapping values are treated as named fields rather than as
    one-entry-per-sample containers. For example:

        {
            "image": Tensor[B, C, H, W],
            "meta": Tensor[B, D],
        }

    must infer B from one of the mapping values rather than using
    len(mapping), which would incorrectly return the number of
    fields.

    Nested mappings, lists, and tuples are handled recursively.

    This function is intentionally NOT a general batch-schema
    validator. It only determines a usable batch-size statistic.

    Raises:
        TrainingError: If neither samples nor targets provides a
            usable batch size, or if a scalar Tensor is encountered.
    """

    for value in (samples, targets):
        size = _try_infer_size_from_value(value)

        if size is not None:
            return size

    raise TrainingError(
        "Unable to determine batch size: neither samples nor targets "
        "is a Tensor, a mapping of batch values, or another sized "
        "container."
    )


def _try_infer_size_from_value(value: Any) -> int | None:
    """
    Attempt to infer batch size from a single value.

    Returns:
        The inferred batch size, or None when this value cannot
        provide a usable size.

    Behavior:

    - Tensor:
        Uses dimension 0.
    - 0-dimensional Tensor:
        Raises TrainingError because it has no batch dimension.
    - Mapping:
        Recursively examines its values.
    - list / tuple:
        Uses the container length.
    - str / bytes:
        Ignored because their length represents characters/bytes,
        not samples.
    - Other sized objects:
        Uses len(value).
    - Unsized objects:
        Returns None.

    This helper deliberately does not validate that multiple fields
    have identical batch dimensions. That responsibility belongs to
    the actual model/data contract, not this statistics helper.
    """

    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            raise TrainingError(
                "Unable to determine batch size: encountered a "
                "scalar (0-dimensional) Tensor, which has no "
                "batch dimension."
            )

        return int(value.shape[0])

    if isinstance(value, Mapping):
        for item in value.values():
            size = _try_infer_size_from_value(item)

            if size is not None:
                return size

        return None

    if isinstance(value, (str, bytes)):
        return None

    if isinstance(value, (list, tuple)):
        return len(value)

    if hasattr(value, "__len__"):
        return len(value)

    return None


def _validate_loss_tensor(loss: Any, *, context: str, require_grad: bool) -> None:
    """
    Validate a criterion's return value. Mirrors trainer.py's helper
    of the same behavior; evaluation always calls this with
    require_grad=False since forward runs under torch.no_grad().
    """

    if not isinstance(loss, torch.Tensor):
        raise TrainingError(
            f"Criterion must return a torch.Tensor during {context}, "
            f"got {type(loss).__name__}."
        )

    if loss.dim() != 0:
        raise TrainingError(
            f"Criterion must return a scalar (0-dimensional) loss "
            f"tensor during {context}, got shape {tuple(loss.shape)}."
        )

    if require_grad and not loss.requires_grad:
        raise TrainingError(
            "Loss tensor does not require grad; it cannot be used "
            "for backpropagation. Ensure model parameters require "
            "grad and the criterion is differentiable."
        )

    loss_value = loss.item()

    if math.isnan(loss_value) or math.isinf(loss_value):
        raise TrainingError(
            f"Loss is not finite during {context} (value={loss_value}); "
            f"aborting rather than continuing with a corrupted run."
        )


class Evaluator:
    """
    Runs local, no-gradient evaluation of a BaseModel over a
    DataLoader.

    Evaluator consumes the existing model and data abstractions; it
    never updates model parameters and never partitions data or
    talks to Flower (see module docstring).
    """

    def __init__(
        self,
        model: BaseModel,
        criterion: Callable[[Any, Any], torch.Tensor],
        metrics: list[Metric] | None = None,
    ) -> None:
        """
        Args:
            model: An already-constructed BaseModel to evaluate.
            criterion: Loss function called as criterion(outputs,
                targets), matching Trainer's contract.
            metrics: Optional list of Metric instances to accumulate
                across the evaluation pass. Each is reset() at the
                start of evaluate() and compute()d at the end. Metric
                names must be unique — duplicate names would silently
                overwrite each other in the returned dict.

        Raises:
            TrainingError: If model/criterion/metrics are invalid, or
                two or more supplied metrics share the same name.
        """

        if not isinstance(model, BaseModel):
            raise TrainingError(
                f"Evaluator requires a BaseModel instance, got "
                f"{type(model).__name__}."
            )

        if not callable(criterion):
            raise TrainingError(
                f"Evaluator requires a callable criterion, got "
                f"{type(criterion).__name__}."
            )

        metrics = metrics if metrics is not None else []

        if not isinstance(metrics, list) or not all(
            isinstance(metric, Metric) for metric in metrics
        ):
            raise TrainingError(
                "Evaluator metrics must be a list of Metric instances."
            )

        metric_names = [metric.name for metric in metrics]

        if len(metric_names) != len(set(metric_names)):
            duplicates = sorted(
                {name for name in metric_names if metric_names.count(name) > 1}
            )
            raise TrainingError(
                f"Evaluator metrics contain duplicate name(s): "
                f"{', '.join(duplicates)}. Each metric must have a "
                f"unique name."
            )

        self._model = model
        self._criterion = criterion
        self._metrics = metrics

    def evaluate(self, dataloader: DataLoader) -> EvaluationResult:
        """
        Run a full no-gradient evaluation pass.

        The model's training/eval mode is restored to whatever it
        was before this call: if the model was already in eval mode,
        it remains in eval mode; if it was in train mode, it is
        returned to train mode afterward. This uses the underlying
        network's `.training` flag rather than unconditionally
        calling train_mode(), which would incorrectly flip a model
        that was already in eval mode. All metrics are reset() at the
        start of every call, so results from a prior evaluate() call
        never leak into this one.

        Args:
            dataloader: A torch.utils.data.DataLoader yielding
                (samples, targets) batches.

        Returns:
            An EvaluationResult summarizing the pass.

        Raises:
            TrainingError: If dataloader is invalid, the DataLoader
                yields no batches, the criterion returns an invalid
                loss (non-scalar or non-finite), or evaluation fails
                for any other reason (original exception chained via
                `from`).
        """

        if not isinstance(dataloader, DataLoader):
            raise TrainingError(
                f"Evaluator.evaluate() requires a "
                f"torch.utils.data.DataLoader, got "
                f"{type(dataloader).__name__}."
            )

        was_training = bool(self._model.network.training)

        logger.info("Evaluation started: device=%s", self._model.device)

        self._model.eval_mode()

        for metric in self._metrics:
            metric.reset()

        total_loss = 0.0
        samples_evaluated = 0
        batches_evaluated = 0

        try:
            with torch.no_grad():
                for batch in dataloader:
                    samples, targets = self._unpack_batch(batch)

                    samples = _move_to_device(samples, self._model.device)
                    targets = _move_to_device(targets, self._model.device)

                    outputs = self._model.forward(samples)
                    loss = self._criterion(outputs, targets)

                    _validate_loss_tensor(
                        loss, context="evaluation", require_grad=False
                    )

                    batch_size = _infer_batch_size(samples, targets)
                    samples_evaluated += batch_size
                    batches_evaluated += 1
                    total_loss += float(loss.item()) * batch_size

                    for metric in self._metrics:
                        metric.update(outputs, targets)

            if batches_evaluated == 0:
                raise TrainingError(
                    "DataLoader produced no batches; cannot evaluate "
                    "on an empty pass."
                )

            mean_loss = total_loss / samples_evaluated
            computed_metrics = {
                metric.name: metric.compute() for metric in self._metrics
            }

        except TrainingError:
            raise
        except Exception as exc:
            raise TrainingError(f"Evaluation failed: {exc}") from exc
        finally:
            if was_training:
                self._model.train_mode()

        logger.info(
            "Evaluation completed: samples=%d, batches=%d, loss=%.6f",
            samples_evaluated,
            batches_evaluated,
            mean_loss,
        )

        return EvaluationResult(
            samples_evaluated=samples_evaluated,
            batches_evaluated=batches_evaluated,
            loss=mean_loss,
            metrics=computed_metrics,
        )

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[Any, Any]:
        """Unpack a DataLoader batch into (samples, targets). Mirrors
        trainer.py's helper of the same behavior."""

        try:
            samples, targets = batch
        except (TypeError, ValueError) as exc:
            raise TrainingError(
                f"Expected each DataLoader batch to unpack into "
                f"(samples, targets); got {exc}."
            ) from exc

        return samples, targets