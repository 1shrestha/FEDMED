"""
Dataset partitioner for FedMed.

This module defines partition_dataset, which deterministically
divides a single validated global FedMedDataset's sample indices
among multiple simulated federated participants.

The partitioner operates strictly above the dataset layer: it never
duplicates FedMedDataset's validation logic, never inspects sample
or target content, and never mutates the global dataset. It only
computes and stores index assignments, then exposes each client's
share of the data through a lightweight, reference-based local view.

This module intentionally does NOT contain:

- Non-IID partitioning strategies (Dirichlet, label/quantity skew)
- DataLoader / batching / sampling
- Training, models, or optimizers
- GPU/device placement
- Flower-specific or networking code
- Real cross-silo data movement (this simulates clients from a
  centralized dataset for development/testing/benchmarking only)
"""

from __future__ import annotations

import numbers
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from src.common.exceptions import DataError
from src.data.dataset import FedMedDataset

IID_STRATEGY = "iid"

SUPPORTED_STRATEGIES: tuple[str, ...] = (IID_STRATEGY,)


class PartitionView(Dataset):
    """
    Lightweight, reference-based local dataset view for one client.

    A PartitionView does not copy or duplicate any sample/target
    data. It stores a reference to the global FedMedDataset and a
    tuple of global indices belonging to this client, and resolves
    ``local_dataset[i]`` to ``global_dataset[indices[i]]`` on access.

    This mirrors the role of ``torch.utils.data.Subset`` but is
    defined locally so FedMed's own abstractions (and DataError
    conventions) are preserved rather than depending on PyTorch's
    subset behavior/error types.
    """

    def __init__(
        self,
        dataset: FedMedDataset,
        indices: tuple[int, ...],
        client_id: str,
    ) -> None:
        """
        Args:
            dataset: The global FedMedDataset this view reads from.
                Held by reference; never copied or mutated.
            indices: Global indices belonging to this client, in
                assignment order.
            client_id: Identifier of the owning client, used in
                error messages and repr().
        """

        self._dataset = dataset
        self._indices = indices
        self._client_id = client_id

    def __len__(self) -> int:
        """Return the number of samples assigned to this client."""
        return len(self._indices)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Resolve a local index to the corresponding global sample.

        Follows the same integer-index contract as
        ``FedMedDataset.__getitem__``: accepts native Python ``int``
        as well as NumPy integer types (e.g. ``numpy.int64``), since
        the IID strategy assigns indices via
        ``numpy.random.default_rng`` / ``numpy.array_split``.
        Negative indices follow standard Python semantics.

        Args:
            index: Local, client-relative integer index.

        Returns:
            The (sample, target) pair from the global dataset at the
            global index this local index maps to.

        Raises:
            DataError: If index is not an integer type (including
                bool, which is rejected despite being an int
                subclass) or is out of range for this client's
                partition.
        """

        if isinstance(index, bool) or not isinstance(index, numbers.Integral):
            raise DataError(
                f"Partition '{self._client_id}': index must be an "
                f"integer, got {type(index).__name__}."
            )

        index = int(index)

        length = len(self._indices)

        if index < -length or index >= length:
            raise DataError(
                f"Partition '{self._client_id}': index {index} out "
                f"of range for partition of size {length}."
            )

        global_index = self._indices[index]

        return self._dataset[global_index]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(client_id={self._client_id!r}, "
            f"size={len(self._indices)})"
        )


@dataclass(frozen=True)
class ClientPartition:
    """
    One client's share of a partitioned global dataset.

    Attributes:
        client_id: Generic identifier, e.g. "client_0". Never encodes
            hospital/site identity — that belongs to later
            configuration/infrastructure layers.
        indices: Global dataset indices assigned to this client, in
            assignment order. Immutable and duplicate-free.
        dataset: A PartitionView resolving local indices into the
            global dataset without copying sample/target data.
        metadata: Non-sensitive, technical partition metadata
            (client_id, num_samples, global_dataset_name, strategy,
            seed). Contains no patient identifiers or clinical
            content.
    """

    client_id: str
    indices: tuple[int, ...]
    dataset: PartitionView
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_dataset(dataset: FedMedDataset) -> None:
    """Ensure the input is an already-constructed FedMedDataset."""

    if not isinstance(dataset, FedMedDataset):
        raise DataError(
            "Partitioner requires a FedMedDataset instance, got "
            f"{type(dataset).__name__}."
        )


def _validate_num_clients(num_clients: int, dataset_size: int) -> None:
    """
    Validate the requested client count against the dataset size.

    Rejects non-int types (including bool, which is a subclass of
    int in Python), non-positive counts, and counts exceeding the
    number of available samples (FedMedDataset never permits empty
    datasets, and every client must receive at least one sample).
    """

    if isinstance(num_clients, bool) or not isinstance(num_clients, int):
        raise DataError(
            "num_clients must be an int, got "
            f"{type(num_clients).__name__}."
        )

    if num_clients <= 0:
        raise DataError(
            f"num_clients must be positive, got {num_clients}."
        )

    if num_clients > dataset_size:
        raise DataError(
            f"num_clients ({num_clients}) cannot exceed dataset "
            f"size ({dataset_size}); every client must receive at "
            f"least one sample."
        )


def _validate_seed(seed: int | None) -> None:
    """Validate the optional seed parameter."""

    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise DataError(
            f"seed must be an int or None, got {type(seed).__name__}."
        )


def _validate_strategy(strategy: str) -> None:
    """Ensure the requested partitioning strategy is supported."""

    if strategy not in SUPPORTED_STRATEGIES:
        raise DataError(
            f"Unsupported partitioning strategy: {strategy!r}. "
            f"Supported strategies: {', '.join(SUPPORTED_STRATEGIES)}."
        )


def _assign_indices_iid(
    dataset_size: int,
    num_clients: int,
    seed: int | None,
) -> list[tuple[int, ...]]:
    """
    Compute a balanced IID assignment of global indices to clients.

    Uses a local numpy Generator (never the global random state) to
    shuffle all global indices, then splits them into num_clients
    contiguous, balanced chunks. Targets are never inspected: this
    strategy only ever operates on index positions, preserving
    compatibility with arbitrary target types (classification,
    regression, segmentation, ...).

    Args:
        dataset_size: Total number of samples in the global dataset.
        num_clients: Number of balanced chunks to produce.
        seed: Seed for the local RNG. None yields a fresh,
            non-reproducible shuffle each call.

    Returns:
        A list of length num_clients, where element i is the tuple
        of global indices assigned to client i. Chunk sizes differ
        by at most 1, sum to dataset_size, and are jointly
        exhaustive and non-overlapping.
    """

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(dataset_size)
    chunks = np.array_split(shuffled_indices, num_clients)

    return [tuple(int(global_index) for global_index in chunk) for chunk in chunks]


_STRATEGIES: dict[str, Callable[[int, int, int | None], list[tuple[int, ...]]]] = {
    IID_STRATEGY: _assign_indices_iid,
}


def partition_dataset(
    dataset: FedMedDataset,
    num_clients: int,
    strategy: str = IID_STRATEGY,
    seed: int | None = None,
) -> dict[str, ClientPartition]:
    """
    Partition a global FedMedDataset's indices among federated clients.

    This simulates multiple federated participants from a single
    centralized dataset for development, testing, and benchmarking.
    It partitions BY INDEX only: no samples or targets are copied,
    reordered, or mutated. The global dataset is left completely
    unchanged.

    Args:
        dataset: A validated, already-constructed FedMedDataset to
            partition.
        num_clients: Number of clients to partition the dataset
            among. Must be a positive int (not bool) not exceeding
            len(dataset), since every client must receive at least
            one sample.
        strategy: Partitioning strategy name. Only "iid" (balanced,
            random, label-agnostic) is currently supported.
        seed: Seed for the local RNG used to shuffle indices before
            assignment. The same (dataset, num_clients, strategy,
            seed) always produces the same assignment. None yields
            an intentionally non-reproducible assignment. Does not
            read or modify numpy's global random state.

    Returns:
        A dict mapping generic client ids ("client_0", "client_1",
        ...) to ClientPartition objects. Every global index in
        range(len(dataset)) appears in exactly one client's indices;
        partition sizes are balanced (max - min <= 1).

    Raises:
        DataError: If dataset, num_clients, strategy, or seed fail
            validation.
    """

    _validate_dataset(dataset)

    dataset_size = len(dataset)

    _validate_num_clients(num_clients, dataset_size)
    _validate_strategy(strategy)
    _validate_seed(seed)

    assign_indices = _STRATEGIES[strategy]
    index_chunks = assign_indices(dataset_size, num_clients, seed)

    dataset_name = dataset.metadata["name"]
    partitions: dict[str, ClientPartition] = {}

    for client_index, indices in enumerate(index_chunks):
        client_id = f"client_{client_index}"

        partitions[client_id] = ClientPartition(
            client_id=client_id,
            indices=indices,
            dataset=PartitionView(dataset, indices, client_id),
            metadata={
                "client_id": client_id,
                "num_samples": len(indices),
                "global_dataset_name": dataset_name,
                "strategy": strategy,
                "seed": seed,
            },
        )

    return partitions