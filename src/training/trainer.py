"""
Local training engine for FedMed.

This module defines Trainer, which runs a standard local PyTorch
training loop over an already-constructed BaseModel and DataLoader.

Trainer is PHASE 2 = LOCAL LEARNING only. It has no knowledge of
federated rounds, clients, hospitals, aggregation, or Flower. A
future Phase 3 Flower client wraps this exact Trainer without
modifying it.

This module intentionally does NOT:

- create datasets, partitions, or DataLoaders
- create or register models (it receives an already-built BaseModel)
- perform federated aggregation or communicate with Flower
- know client/server/hospital identity
- implement evaluation logic (see evaluator.py)
- assume a specific task (classification/regression/segmentation) or
  a specific loss function

Loss aggregation contract:

    The criterion is expected to return a scalar tensor representing
    the MEAN loss over the current batch (the default reduction mode
    for standard PyTorch losses, e.g. reduction="mean"). Because
    batches may have unequal sizes (the final batch of an epoch is
    often smaller), Trainer aggregates epoch loss as a SAMPLE-WEIGHTED
    mean, not a simple average of per-batch means:

        total_loss   += batch_loss * batch_size
        total_samples += batch_size
        epoch_mean_loss = total_loss / total_samples

    A criterion using a different reduction (e.g. "sum") will still
    run, but the resulting epoch_losses will not represent a
    sample-weighted mean under that reduction; Trainer does not
    detect or adapt to the criterion's reduction mode.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.common.config import TrainingConfig
from src.common.exceptions import TrainingError
from src.common.logging import get_logger
from src.models.base_model import BaseModel

logger = get_logger(__name__)


_SUPPORTED_OPTIMIZERS: dict[str, type[Optimizer]] = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}


@dataclass(frozen=True)
class TrainingResult:
    """
    Structured outcome of a local Trainer.train() call.

    Deliberately contains only local-training facts. No client_id,
    server_id, round number, aggregation weight, or Flower
    parameters — those are Phase 3 concerns layered on top of this
    result, not part of it.

    Attributes:
        epochs_completed: Number of local epochs actually run.
        samples_processed: Total number of samples seen across all
            epochs. Tracked explicitly because a future federated
            aggregator needs local sample counts and this is the
            layer that actually observes them batch by batch.
        batches_processed: Total number of optimizer steps taken
            across all epochs.
        epoch_losses: Sample-weighted mean training loss for each
            completed epoch, in order.
        final_loss: The last entry of epoch_losses.
    """

    epochs_completed: int
    samples_processed: int
    batches_processed: int
    epoch_losses: list[float] = field(default_factory=list)
    final_loss: float = float("nan")


def _move_to_device(value: Any, device: torch.device) -> Any:
    """
    Recursively move tensor-containing batch structures to a device.

    Supports the standard PyTorch batch containers produced by
    default_collate: Tensor, mappings, lists, and tuples.

    Structure is preserved. Nothing is mutated in place, and
    non-tensor leaf values are passed through unchanged.

    This is a best-effort transfer helper, not a schema validator.
    """

    if isinstance(value, torch.Tensor):
        return value.to(device)

    if isinstance(value, Mapping):
        return {
            key: _move_to_device(item, device)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _move_to_device(item, device)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _move_to_device(item, device)
            for item in value
        )

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


def _validate_loss_tensor(
    loss: Any,
    *,
    context: str,
    require_grad: bool,
) -> None:
    """
    Validate a criterion's return value before it participates in
    backpropagation or is merely recorded.

    Raises:
        TrainingError: If loss is not a torch.Tensor, is not scalar,
            does not require grad when required, or is NaN/infinite.
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


def _validate_optimizer_model_compatibility(
    optimizer: Optimizer,
    model: BaseModel,
) -> None:
    """
    Verify a caller-supplied optimizer is actually associated with
    the supplied model's parameters.

    Compares parameter tensor objects by identity rather than value.

    At least one optimizer parameter must belong to the supplied
    model. This allows legitimate architectures where some model
    parameters are intentionally frozen and therefore excluded from
    the optimizer.

    Raises:
        TrainingError: If the optimizer contains no parameters
            belonging to the supplied model.
    """

    model_parameter_ids = {
        id(parameter)
        for parameter in model.network.parameters()
    }

    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    if model_parameter_ids.isdisjoint(optimizer_parameter_ids):
        raise TrainingError(
            "The supplied optimizer does not contain any parameters "
            "belonging to the supplied model; it appears to have been "
            "constructed for a different model instance."
        )


def _validate_training_config_type(config: Any) -> None:
    """Ensure config is an actual TrainingConfig instance."""

    if not isinstance(config, TrainingConfig):
        raise TrainingError(
            f"Trainer config must be a TrainingConfig instance, got "
            f"{type(config).__name__}."
        )


def _validate_local_epochs_field(local_epochs: Any) -> None:
    """Validate TrainingConfig.local_epochs as a positive int."""

    if isinstance(local_epochs, bool) or not isinstance(local_epochs, int):
        raise TrainingError(
            f"TrainingConfig.local_epochs must be an int, got "
            f"{type(local_epochs).__name__}."
        )

    if local_epochs <= 0:
        raise TrainingError(
            f"TrainingConfig.local_epochs must be positive, got "
            f"{local_epochs}."
        )


def _validate_learning_rate_field(learning_rate: Any) -> None:
    """Validate TrainingConfig.learning_rate as finite and > 0."""

    if isinstance(learning_rate, bool) or not isinstance(
        learning_rate,
        (int, float),
    ):
        raise TrainingError(
            f"TrainingConfig.learning_rate must be numeric, got "
            f"{type(learning_rate).__name__}."
        )

    value = float(learning_rate)

    if math.isnan(value) or math.isinf(value):
        raise TrainingError(
            f"TrainingConfig.learning_rate must be finite, got "
            f"{learning_rate}."
        )

    if value <= 0:
        raise TrainingError(
            f"TrainingConfig.learning_rate must be strictly positive, "
            f"got {learning_rate}."
        )


def _validate_optimizer_name_field(optimizer_name: Any) -> None:
    """Validate TrainingConfig.optimizer as a non-empty string."""

    if not isinstance(optimizer_name, str) or not optimizer_name.strip():
        raise TrainingError(
            "TrainingConfig.optimizer must be a non-empty string."
        )


class Trainer:
    """
    Runs local training of a BaseModel over a DataLoader.

    Trainer consumes the existing model, data, and configuration
    abstractions; it does not create or duplicate any of them.

    It uses model.network.parameters() only to construct an optimizer
    when one is not supplied, and otherwise interacts with the model
    through the BaseModel interface.
    """

    def __init__(
        self,
        model: BaseModel,
        criterion: Callable[[Any, Any], torch.Tensor],
        optimizer: Optimizer | None = None,
        config: TrainingConfig | None = None,
    ) -> None:
        """
        Args:
            model: An already-constructed BaseModel.
            criterion: Loss function called as
                criterion(outputs, targets).
            optimizer: A pre-constructed optimizer over model
                parameters. If omitted, one is constructed from
                TrainingConfig.
            config: TrainingConfig providing optimizer,
                learning_rate, and local_epochs where applicable.

        Raises:
            TrainingError: If model, criterion, optimizer, or config
                are invalid.
        """

        if not isinstance(model, BaseModel):
            raise TrainingError(
                f"Trainer requires a BaseModel instance, got "
                f"{type(model).__name__}."
            )

        if not callable(criterion):
            raise TrainingError(
                f"Trainer requires a callable criterion, got "
                f"{type(criterion).__name__}."
            )

        if optimizer is not None and not isinstance(
            optimizer,
            Optimizer,
        ):
            raise TrainingError(
                f"Trainer optimizer must be a torch.optim.Optimizer "
                f"instance, got {type(optimizer).__name__}."
            )

        if optimizer is None and config is None:
            raise TrainingError(
                "Trainer requires either an explicit optimizer or a "
                "TrainingConfig to build one from."
            )

        if config is not None:
            _validate_training_config_type(config)
            _validate_local_epochs_field(config.local_epochs)

        if optimizer is not None:
            _validate_optimizer_model_compatibility(
                optimizer,
                model,
            )

        self._model = model
        self._criterion = criterion
        self._config = config

        self._optimizer = (
            optimizer
            if optimizer is not None
            else self._build_optimizer(
                model.network.parameters(),
                config,
            )
        )

    @staticmethod
    def _build_optimizer(
        parameters: Any,
        config: TrainingConfig,
    ) -> Optimizer:
        """
        Construct an optimizer from TrainingConfig.

        Raises:
            TrainingError: If configuration is invalid, the optimizer
                is unsupported, or construction fails.
        """

        _validate_optimizer_name_field(config.optimizer)
        _validate_learning_rate_field(config.learning_rate)

        optimizer_name = config.optimizer.strip().lower()

        if optimizer_name not in _SUPPORTED_OPTIMIZERS:
            supported = ", ".join(
                sorted(_SUPPORTED_OPTIMIZERS)
            )

            raise TrainingError(
                f"Unsupported optimizer '{config.optimizer}'. "
                f"Supported optimizers: {supported}."
            )

        optimizer_class = _SUPPORTED_OPTIMIZERS[optimizer_name]

        try:
            return optimizer_class(
                parameters,
                lr=float(config.learning_rate),
            )
        except Exception as exc:
            raise TrainingError(
                f"Failed to construct optimizer '{optimizer_name}': {exc}"
            ) from exc

    def train(
        self,
        dataloader: DataLoader,
        epochs: int | None = None,
    ) -> TrainingResult:
        """
        Run local training for the requested number of epochs.

        Standard PyTorch lifecycle:

            model.train_mode()
            zero_grad()
            forward()
            criterion()
            validate loss
            backward()
            optimizer.step()

        Epoch loss is aggregated as a sample-weighted mean.

        Raises:
            TrainingError: If the DataLoader, epoch count, loss,
                batch structure, or training operation is invalid.
        """

        if not isinstance(dataloader, DataLoader):
            raise TrainingError(
                f"Trainer.train() requires a torch.utils.data.DataLoader, "
                f"got {type(dataloader).__name__}."
            )

        resolved_epochs = self._resolve_epochs(epochs)

        logger.info(
            "Training started: epochs=%s, device=%s",
            resolved_epochs,
            self._model.device,
        )

        self._model.train_mode()

        epoch_losses: list[float] = []
        samples_processed = 0
        batches_processed = 0

        try:
            for epoch_index in range(resolved_epochs):
                epoch_loss_total = 0.0
                epoch_samples = 0
                epoch_batch_count = 0

                for batch in dataloader:
                    samples, targets = self._unpack_batch(batch)

                    samples = _move_to_device(
                        samples,
                        self._model.device,
                    )

                    targets = _move_to_device(
                        targets,
                        self._model.device,
                    )

                    self._optimizer.zero_grad()

                    outputs = self._model.forward(samples)

                    loss = self._criterion(
                        outputs,
                        targets,
                    )

                    _validate_loss_tensor(
                        loss,
                        context="training",
                        require_grad=True,
                    )

                    loss.backward()
                    self._optimizer.step()

                    batch_size = _infer_batch_size(
                        samples,
                        targets,
                    )

                    samples_processed += batch_size
                    batches_processed += 1

                    epoch_batch_count += 1
                    epoch_samples += batch_size

                    epoch_loss_total += (
                        float(loss.item()) * batch_size
                    )

                if epoch_batch_count == 0:
                    raise TrainingError(
                        "DataLoader produced no batches; cannot train "
                        "on an empty epoch."
                    )

                epoch_mean_loss = (
                    epoch_loss_total / epoch_samples
                )

                epoch_losses.append(epoch_mean_loss)

                logger.info(
                    "Epoch %d/%d completed: mean_loss=%.6f",
                    epoch_index + 1,
                    resolved_epochs,
                    epoch_mean_loss,
                )

        except TrainingError:
            raise

        except Exception as exc:
            raise TrainingError(
                f"Training failed: {exc}"
            ) from exc

        return TrainingResult(
            epochs_completed=resolved_epochs,
            samples_processed=samples_processed,
            batches_processed=batches_processed,
            epoch_losses=epoch_losses,
            final_loss=epoch_losses[-1],
        )

    def _resolve_epochs(
        self,
        epochs: int | None,
    ) -> int:
        """
        Determine the effective epoch count and validate it.
        """

        if epochs is None:
            if self._config is None:
                raise TrainingError(
                    "epochs must be provided explicitly when Trainer "
                    "was constructed without a TrainingConfig."
                )

            epochs = self._config.local_epochs

        if isinstance(epochs, bool) or not isinstance(
            epochs,
            int,
        ):
            raise TrainingError(
                f"epochs must be an int, got {type(epochs).__name__}."
            )

        if epochs <= 0:
            raise TrainingError(
                f"epochs must be positive, got {epochs}."
            )

        return epochs

    @staticmethod
    def _unpack_batch(
        batch: Any,
    ) -> tuple[Any, Any]:
        """
        Unpack a DataLoader batch into (samples, targets).

        FedMedDataset and PartitionView yield (sample, target) pairs,
        so the standard PyTorch collation mechanism produces a
        two-element (samples_batch, targets_batch) structure.

        The structure is validated explicitly so malformed
        DataLoader output produces a clear TrainingError.
        """

        try:
            samples, targets = batch

        except (TypeError, ValueError) as exc:
            raise TrainingError(
                "Expected each DataLoader batch to unpack into "
                f"(samples, targets); got {exc}."
            ) from exc

        return samples, targets