"""
Tests for the FedMed DataLoader boundary (create_dataloader).

This module tests:

- Valid DataLoader construction with default and custom arguments
- Compatibility with both FedMedDataset and PartitionView inputs
- Correct batch counts (with and without drop_last), including
  batch_size larger than the dataset and batch_size == 1
- Sample/target pairing is preserved through batching
- shuffle does not mutate PartitionView indices or the global dataset
- shuffle preserves complete, non-duplicated sample coverage
- persistent_workers/num_workers dependency enforcement
- NumPy integer acceptance for batch_size / num_workers
- Generic (non-classification) target types: regression floats and
  vector targets, proving the loader makes no classification-label
  assumption
- Rejection of invalid dataset, batch_size, num_workers, and
  boolean-flag inputs
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.common.exceptions import DataError
from src.data.dataset import FedMedDataset
from src.data.loader import create_dataloader
from src.data.partitioner import ClientPartition, PartitionView, partition_dataset


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def dataset() -> FedMedDataset:
    """A generic, non-medical 100-sample dataset for loader tests."""

    samples = list(range(100))
    targets = [index % 2 for index in range(100)]

    return FedMedDataset(
        samples=samples,
        targets=targets,
        name="phase1_dataset",
    )


@pytest.fixture
def client_partition(dataset: FedMedDataset) -> ClientPartition:
    """A single client's ClientPartition, as produced by the partitioner."""

    partitions = partition_dataset(dataset, num_clients=5, seed=42)

    return partitions["client_0"]


# ============================================================
# Valid construction
# ============================================================


def test_create_dataloader_default_arguments(dataset: FedMedDataset) -> None:
    """Verify a DataLoader is constructed with sensible defaults."""

    loader = create_dataloader(dataset)

    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 32
    assert loader.num_workers == 0


def test_create_dataloader_custom_batch_size(dataset: FedMedDataset) -> None:
    """Verify batch_size is applied and produces the expected batch shape."""

    loader = create_dataloader(dataset, batch_size=10)

    first_batch_samples, first_batch_targets = next(iter(loader))

    assert first_batch_samples.shape[0] == 10
    assert first_batch_targets.shape[0] == 10


def test_create_dataloader_accepts_fedmed_dataset(dataset: FedMedDataset) -> None:
    """A plain FedMedDataset must be accepted directly."""

    loader = create_dataloader(dataset, batch_size=25)

    batches = list(loader)

    assert len(batches) == 4


def test_create_dataloader_accepts_partition_view(
    client_partition: ClientPartition,
) -> None:
    """
    A client's PartitionView (from ClientPartition.dataset) must be
    accepted directly, without reconstructing a new dataset from
    samples/targets.
    """

    loader = create_dataloader(client_partition.dataset, batch_size=4)

    total_samples = sum(batch[0].shape[0] for batch in loader)

    assert total_samples == len(client_partition.indices)


# ============================================================
# Batch counts / drop_last / batch_size edge cases
# ============================================================


def test_batch_count_without_drop_last(dataset: FedMedDataset) -> None:
    """100 samples, batch_size=30, drop_last=False -> 4 batches (last partial)."""

    loader = create_dataloader(dataset, batch_size=30, drop_last=False)

    batches = list(loader)

    assert len(batches) == 4
    assert batches[-1][0].shape[0] == 10


def test_batch_count_with_drop_last(dataset: FedMedDataset) -> None:
    """100 samples, batch_size=30, drop_last=True -> 3 full batches."""

    loader = create_dataloader(dataset, batch_size=30, drop_last=True)

    batches = list(loader)

    assert len(batches) == 3
    assert all(batch[0].shape[0] == 30 for batch in batches)


def test_batch_size_larger_than_dataset_without_drop_last(
    dataset: FedMedDataset,
) -> None:
    """
    batch_size larger than the dataset size must remain valid: it
    yields a single batch containing every sample, not an error.
    """

    loader = create_dataloader(dataset, batch_size=500, drop_last=False)

    batches = list(loader)

    assert len(batches) == 1
    assert batches[0][0].shape[0] == 100


def test_batch_size_larger_than_dataset_with_drop_last_yields_no_batches(
    dataset: FedMedDataset,
) -> None:
    """
    batch_size larger than the dataset size with drop_last=True must
    naturally yield zero batches, matching PyTorch's own behavior.
    """

    loader = create_dataloader(dataset, batch_size=500, drop_last=True)

    batches = list(loader)

    assert len(batches) == 0


def test_batch_size_one_produces_single_sample_batches(
    dataset: FedMedDataset,
) -> None:
    """batch_size=1 must produce one batch per sample."""

    loader = create_dataloader(dataset, batch_size=1)

    batches = list(loader)

    assert len(batches) == 100
    assert all(batch[0].shape[0] == 1 for batch in batches)


def test_partitioner_and_loader_integration(dataset: FedMedDataset) -> None:
    """100 samples over 5 clients, batch_size=4 -> 5 batches per client loader."""

    partitions = partition_dataset(dataset, num_clients=5, seed=42)

    for partition in partitions.values():
        loader = create_dataloader(partition.dataset, batch_size=4)
        batches = list(loader)

        assert len(batches) == 5
        assert sum(batch[0].shape[0] for batch in batches) == 20


# ============================================================
# Sample/target pairing
# ============================================================


def test_sample_target_pairing_preserved_after_batching(
    dataset: FedMedDataset,
) -> None:
    """
    Batching must not break the sample/target pairing established by
    FedMedDataset. For this fixture, target == sample % 2 for every
    item, regardless of batch boundaries.
    """

    loader = create_dataloader(dataset, batch_size=10, shuffle=False)

    for batch_samples, batch_targets in loader:
        for sample, target in zip(batch_samples.tolist(), batch_targets.tolist()):
            assert target == sample % 2


# ============================================================
# Generic (non-classification) target types
# ============================================================


def test_regression_float_targets_are_batched_correctly() -> None:
    """
    FedMedDataset does not assume integer classification labels.
    Verify the loader batches scalar regression targets correctly,
    with values preserved and correctly paired to their samples.

    Samples/targets are represented as per-item torch tensors here
    (rather than raw nested Python lists) so PyTorch's default
    collation produces a single rectangular batch tensor per field;
    this is a data-representation choice for the test, not a loader
    restriction — loader.py imposes no target-type assumption either
    way.
    """

    samples = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 4.0]),
        torch.tensor([5.0, 6.0]),
    ]
    targets = [
        torch.tensor(0.5),
        torch.tensor(1.5),
        torch.tensor(2.5),
    ]

    regression_dataset = FedMedDataset(
        samples=samples,
        targets=targets,
        name="regression_loader_test",
    )

    loader = create_dataloader(regression_dataset, batch_size=2, shuffle=False)

    batches = list(loader)

    assert len(batches) == 2

    all_samples = torch.cat([batch[0] for batch in batches], dim=0)
    all_targets = torch.cat([batch[1] for batch in batches], dim=0)

    assert all_samples.shape == (3, 2)
    assert all_targets.shape == (3,)
    assert torch.equal(all_targets, torch.tensor([0.5, 1.5, 2.5]))


def test_vector_targets_are_batched_correctly() -> None:
    """
    Verify the loader correctly batches vector-valued targets (e.g.
    multi-label or one-hot-style targets), confirming compatibility
    with future classification-vector, multi-label, or
    segmentation-style-metadata tasks without any task-specific
    logic in loader.py.
    """

    samples = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 4.0]),
        torch.tensor([5.0, 6.0]),
    ]
    targets = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 1.0]),
    ]

    vector_target_dataset = FedMedDataset(
        samples=samples,
        targets=targets,
        name="vector_target_loader_test",
    )

    loader = create_dataloader(vector_target_dataset, batch_size=2, shuffle=False)

    batches = list(loader)

    assert len(batches) == 2

    all_samples = torch.cat([batch[0] for batch in batches], dim=0)
    all_targets = torch.cat([batch[1] for batch in batches], dim=0)

    assert all_samples.shape == (3, 2)
    assert all_targets.shape == (3, 2)
    assert torch.equal(
        all_targets,
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    )


# ============================================================
# shuffle: no mutation, complete coverage, no duplication
# ============================================================


def test_shuffle_does_not_mutate_partition_indices(
    dataset: FedMedDataset, client_partition: ClientPartition
) -> None:
    """
    shuffle=True must only affect batch draw order; it must never
    reorder or mutate the underlying PartitionView's fixed indices.
    """

    indices_before = tuple(client_partition.indices)

    loader = create_dataloader(
        client_partition.dataset, batch_size=4, shuffle=True
    )
    list(loader)

    assert tuple(client_partition.indices) == indices_before


def test_shuffle_does_not_mutate_global_dataset(dataset: FedMedDataset) -> None:
    """shuffle=True on the global dataset must not mutate it."""

    item_before = dataset[0]

    loader = create_dataloader(dataset, batch_size=10, shuffle=True)
    list(loader)

    assert dataset[0] == item_before


def test_shuffle_preserves_complete_sample_coverage(dataset: FedMedDataset) -> None:
    """
    shuffle=True must not lose or duplicate samples: every sample must
    appear exactly once across all batches, regardless of order. This
    single check jointly proves no loss, no duplication, and complete
    coverage.
    """

    loader = create_dataloader(dataset, batch_size=10, shuffle=True)

    observed_samples = []
    for batch_samples, _ in loader:
        observed_samples.extend(batch_samples.tolist())

    assert len(observed_samples) == 100
    assert len(set(observed_samples)) == 100
    assert sorted(observed_samples) == list(range(100))


# ============================================================
# persistent_workers / num_workers dependency
# ============================================================


def test_persistent_workers_requires_positive_num_workers(
    dataset: FedMedDataset,
) -> None:
    """persistent_workers=True with num_workers=0 must be rejected."""

    with pytest.raises(DataError):
        create_dataloader(
            dataset, num_workers=0, persistent_workers=True
        )


def test_persistent_workers_allowed_with_positive_num_workers(
    dataset: FedMedDataset,
) -> None:
    """
    persistent_workers=True with num_workers>0 must be accepted. Only
    construction and the resulting configuration are verified here;
    worker process lifecycle is left to DataLoader itself.
    """

    loader = create_dataloader(
        dataset, num_workers=1, persistent_workers=True
    )

    assert loader.persistent_workers is True


# ============================================================
# Invalid dataset input
# ============================================================


def test_invalid_dataset_type_rejected() -> None:
    """A raw list must not be silently accepted as a dataset."""

    with pytest.raises(DataError):
        create_dataloader([1, 2, 3])  # type: ignore[arg-type]


def test_empty_partition_view_rejected(dataset: FedMedDataset) -> None:
    """
    A PartitionView with no indices must be rejected defensively.
    This case cannot arise from partition_dataset() itself (which
    guarantees non-empty partitions), so it is exercised here via
    direct PartitionView construction.
    """

    empty_view = PartitionView(dataset, indices=(), client_id="client_empty")

    with pytest.raises(DataError):
        create_dataloader(empty_view)


# ============================================================
# Invalid batch_size
# ============================================================


def test_zero_batch_size_rejected(dataset: FedMedDataset) -> None:
    with pytest.raises(DataError):
        create_dataloader(dataset, batch_size=0)


def test_negative_batch_size_rejected(dataset: FedMedDataset) -> None:
    with pytest.raises(DataError):
        create_dataloader(dataset, batch_size=-1)


def test_float_batch_size_rejected(dataset: FedMedDataset) -> None:
    with pytest.raises(DataError):
        create_dataloader(dataset, batch_size=1.5)


def test_string_batch_size_rejected(dataset: FedMedDataset) -> None:
    with pytest.raises(DataError):
        create_dataloader(dataset, batch_size="32")


def test_none_batch_size_rejected(dataset: FedMedDataset) -> None:
    with pytest.raises(DataError):
        create_dataloader(dataset, batch_size=None)


def test_bool_batch_size_rejected(dataset: FedMedDataset) -> None:
    with pytest.raises(DataError):
        create_dataloader(dataset, batch_size=True)


def test_numpy_integer_batch_size_accepted(dataset: FedMedDataset) -> None:
    """NumPy integer types (e.g. np.int64) must be accepted for batch_size."""

    loader = create_dataloader(dataset, batch_size=np.int64(16))

    assert loader.batch_size == 16


# ============================================================
# Invalid num_workers
# ============================================================


def test_negative_num_workers_rejected(dataset: FedMedDataset) -> None:
    with pytest.raises(DataError):
        create_dataloader(dataset, num_workers=-1)


def test_float_num_workers_rejected(dataset: FedMedDataset) -> None:
    with pytest.raises(DataError):
        create_dataloader(dataset, num_workers=1.5)


def test_string_num_workers_rejected(dataset: FedMedDataset) -> None:
    with pytest.raises(DataError):
        create_dataloader(dataset, num_workers="2")


def test_none_num_workers_rejected(dataset: FedMedDataset) -> None:
    with pytest.raises(DataError):
        create_dataloader(dataset, num_workers=None)


def test_bool_num_workers_rejected(dataset: FedMedDataset) -> None:
    with pytest.raises(DataError):
        create_dataloader(dataset, num_workers=True)


def test_numpy_integer_num_workers_accepted(dataset: FedMedDataset) -> None:
    """NumPy integer types (e.g. np.int64) must be accepted for num_workers."""

    loader = create_dataloader(dataset, num_workers=np.int64(0))

    assert loader.num_workers == 0


# ============================================================
# Invalid boolean flags
# ============================================================


def test_non_bool_shuffle_rejected(dataset: FedMedDataset) -> None:
    with pytest.raises(DataError):
        create_dataloader(dataset, shuffle=1)


def test_non_bool_drop_last_rejected(dataset: FedMedDataset) -> None:
    with pytest.raises(DataError):
        create_dataloader(dataset, drop_last="yes")


def test_non_bool_pin_memory_rejected(dataset: FedMedDataset) -> None:
    with pytest.raises(DataError):
        create_dataloader(dataset, pin_memory=0)


def test_non_bool_persistent_workers_rejected(dataset: FedMedDataset) -> None:
    with pytest.raises(DataError):
        create_dataloader(dataset, num_workers=1, persistent_workers="true")