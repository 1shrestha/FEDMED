"""
DataLoader construction for FedMed.

This module defines create_dataloader, the boundary between FedMed's
own map-style data abstractions (FedMedDataset, PartitionView) and
PyTorch's torch.utils.data.DataLoader batching mechanism.

loader.py intentionally does NOT:

- create or partition datasets (that is dataset.py / partitioner.py)
- implement custom batching, sampling, or collation
- perform device/GPU transfer (that belongs to trainer.py)
- reshuffle or mutate partition indices (partitioner.py owns WHICH
  samples a client has; this module only controls the ORDER batches
  are drawn in, via PyTorch's own shuffle mechanism)
- know or care whether a dataset originated from a federated
  partition; it simply receives a map-style dataset
"""

from __future__ import annotations

import numbers
from typing import Any

from torch.utils.data import DataLoader

from src.common.exceptions import DataError
from src.data.dataset import FedMedDataset
from src.data.partitioner import PartitionView


DEFAULT_BATCH_SIZE = 32
DEFAULT_SHUFFLE = False
DEFAULT_DROP_LAST = False
DEFAULT_NUM_WORKERS = 0
DEFAULT_PIN_MEMORY = False
DEFAULT_PERSISTENT_WORKERS = False

_ACCEPTED_DATASET_TYPES = (FedMedDataset, PartitionView)


def _validate_dataset(dataset: Any) -> None:
    """
    Ensure the input is a FedMed map-style dataset abstraction.

    Only FedMedDataset (a full global dataset) and PartitionView (a
    client's index-based view over a global dataset, as produced by
    partition_dataset) are accepted. Arbitrary raw arrays or generic
    objects are rejected so this module never becomes a second place
    that silently constructs or reinterprets FedMed data.

    The parameter is typed as Any because this function's job is
    precisely to safely inspect and reject values of unknown/invalid
    runtime type before any narrower type can be assumed.
    """

    if not isinstance(dataset, _ACCEPTED_DATASET_TYPES):
        raise DataError(
            "create_dataloader requires a FedMedDataset or "
            f"PartitionView instance, got {type(dataset).__name__}."
        )

    if len(dataset) == 0:
        raise DataError(
            "create_dataloader requires a non-empty dataset."
        )


def _validate_batch_size(batch_size: int) -> None:
    """
    Validate batch_size as a positive integer.

    Rejects bool (a subclass of int, but not a valid configuration
    value here), float, str, None, and non-positive values. Accepts
    NumPy integer types via numbers.Integral, consistent with the
    integer-handling convention established by dataset.py and
    partitioner.py.
    """

    if isinstance(batch_size, bool) or not isinstance(
        batch_size, numbers.Integral
    ):
        raise DataError(
            f"batch_size must be an integer, "
            f"got {type(batch_size).__name__}."
        )

    if int(batch_size) <= 0:
        raise DataError(
            f"batch_size must be positive, got {int(batch_size)}."
        )


def _validate_num_workers(num_workers: int) -> None:
    """
    Validate num_workers as a non-negative integer.

    Rejects bool, float, str, None, and negative values. Accepts
    NumPy integer types via numbers.Integral.
    """

    if isinstance(num_workers, bool) or not isinstance(
        num_workers, numbers.Integral
    ):
        raise DataError(
            f"num_workers must be an integer, "
            f"got {type(num_workers).__name__}."
        )

    if int(num_workers) < 0:
        raise DataError(
            f"num_workers must be non-negative, got {int(num_workers)}."
        )


def _validate_bool_flag(value: bool, name: str) -> None:
    """Validate that a loader configuration flag is an actual bool."""

    if not isinstance(value, bool):
        raise DataError(
            f"{name} must be a bool, got {type(value).__name__}."
        )


def _validate_persistent_workers_dependency(
    persistent_workers: bool,
    num_workers: int,
) -> None:
    """
    Enforce PyTorch's persistent_workers/num_workers dependency
    explicitly, with a clear FedMed-level error, rather than letting
    a less legible error surface from inside DataLoader construction.
    """

    if persistent_workers and int(num_workers) == 0:
        raise DataError(
            "persistent_workers=True requires num_workers > 0; "
            "got num_workers=0."
        )


def create_dataloader(
    dataset: FedMedDataset | PartitionView,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = DEFAULT_SHUFFLE,
    drop_last: bool = DEFAULT_DROP_LAST,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool = DEFAULT_PIN_MEMORY,
    persistent_workers: bool = DEFAULT_PERSISTENT_WORKERS,
) -> DataLoader:
    """
    Construct a validated torch.utils.data.DataLoader over a FedMed
    map-style dataset (FedMedDataset or PartitionView).

    This function configures and validates PyTorch's own DataLoader;
    it does not implement batching, sampling, or collation itself,
    and it never inspects or copies sample/target content. shuffle
    controls only the ORDER in which a dataset's existing samples
    are drawn into batches — it never reorders or mutates a
    PartitionView's underlying indices, which are fixed by
    partition_dataset.

    batch_size may exceed the dataset size; with drop_last=False
    this simply yields a single, smaller-than-usual batch, and with
    drop_last=True it yields zero batches, matching PyTorch's own
    DataLoader semantics.

    Targets are never inspected or assumed to be classification
    labels. They are passed to PyTorch's DataLoader, whose collation
    mechanism is responsible for constructing the batch
    representation.

    Args:
        dataset:
            A FedMedDataset or a client PartitionView (as returned
            by ClientPartition.dataset from
            src.data.partitioner.partition_dataset).

        batch_size:
            Positive number of samples per batch.

        shuffle:
            Whether to draw batches in shuffled order each epoch.
            Does not affect which samples the dataset contains.

        drop_last:
            Whether to drop the final incomplete batch when the
            dataset size is not evenly divisible by batch_size.

        num_workers:
            Number of subprocess workers for data loading.
            0 loads in the main process (safe default for Colab,
            Windows, and Docker).

        pin_memory:
            Whether to pin batch tensors in host memory for faster
            host-to-device transfer. Does NOT move data to a device;
            actual device transfer belongs to trainer.py.

        persistent_workers:
            Whether worker processes persist across epochs.
            Requires num_workers > 0.

    Returns:
        A configured torch.utils.data.DataLoader over the given
        dataset.

    Raises:
        DataError:
            If dataset, batch_size, num_workers, any boolean flag,
            or the persistent_workers/num_workers dependency fail
            validation.
    """

    _validate_dataset(dataset)
    _validate_batch_size(batch_size)
    _validate_num_workers(num_workers)
    _validate_bool_flag(shuffle, "shuffle")
    _validate_bool_flag(drop_last, "drop_last")
    _validate_bool_flag(pin_memory, "pin_memory")
    _validate_bool_flag(persistent_workers, "persistent_workers")
    _validate_persistent_workers_dependency(
        persistent_workers,
        num_workers,
    )

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )