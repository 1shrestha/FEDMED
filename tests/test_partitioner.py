"""
Tests for the FedMed dataset partitioner.

This module tests:

- partition_dataset construction and single/multi-client behavior
- Balanced IID sizing (divisible and non-divisible dataset sizes)
- Partition invariants: complete coverage, no overlap, no
  duplicates, correct total count, non-empty partitions
- Deterministic partitioning under a fixed seed, and differing
  behavior across seeds
- Original dataset immutability after partitioning
- numpy global random state isolation
- PartitionView index contract (int, numpy int, negative, invalid)
- ClientPartition metadata correctness
- Rejection of invalid num_clients, dataset, and strategy inputs
- Partition dataset view retrieval against the global dataset
"""

import numpy as np
import pytest

from src.common.exceptions import DataError
from src.data.dataset import FedMedDataset
from src.data.partitioner import ClientPartition, PartitionView, partition_dataset


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def dataset() -> FedMedDataset:
    """A generic, non-medical 100-sample dataset for partition tests."""

    samples = list(range(100))
    targets = [index % 2 for index in range(100)]

    return FedMedDataset(
        samples=samples,
        targets=targets,
        name="phase1_dataset",
    )


@pytest.fixture
def small_dataset() -> FedMedDataset:
    """A 10-sample dataset for exercising non-divisible partitioning."""

    samples = list(range(10))
    targets = [index % 2 for index in range(10)]

    return FedMedDataset(
        samples=samples,
        targets=targets,
        name="small_dataset",
    )


@pytest.fixture
def partition_view(dataset: FedMedDataset) -> PartitionView:
    """A PartitionView over three known global indices."""

    return PartitionView(dataset, indices=(5, 10, 15), client_id="client_test")


# ============================================================
# Valid construction / basic behavior
# ============================================================


def test_partition_dataset_constructs_successfully(dataset: FedMedDataset) -> None:
    """Verify a valid partitioning call returns client partitions."""

    partitions = partition_dataset(dataset, num_clients=5, seed=42)

    assert len(partitions) == 5
    assert all(isinstance(p, ClientPartition) for p in partitions.values())


def test_single_client_receives_all_samples(dataset: FedMedDataset) -> None:
    """A single client must receive every sample in the dataset."""

    partitions = partition_dataset(dataset, num_clients=1, seed=42)

    assert set(partitions.keys()) == {"client_0"}
    assert len(partitions["client_0"].indices) == len(dataset)
    assert sorted(partitions["client_0"].indices) == list(range(len(dataset)))


def test_multiple_clients_partition(dataset: FedMedDataset) -> None:
    """Multiple clients each receive a non-trivial share of indices."""

    partitions = partition_dataset(dataset, num_clients=4, seed=42)

    assert set(partitions.keys()) == {
        "client_0", "client_1", "client_2", "client_3",
    }


# ============================================================
# Balanced sizing
# ============================================================


def test_balanced_partition_when_divisible(dataset: FedMedDataset) -> None:
    """100 samples over 5 clients must split evenly into 20 each."""

    partitions = partition_dataset(dataset, num_clients=5, seed=42)

    sizes = [len(p.indices) for p in partitions.values()]

    assert sizes == [20, 20, 20, 20, 20]


def test_balanced_partition_when_not_divisible(small_dataset: FedMedDataset) -> None:
    """10 samples over 3 clients must split as close to even as possible."""

    partitions = partition_dataset(small_dataset, num_clients=3, seed=42)

    sizes = sorted((len(p.indices) for p in partitions.values()), reverse=True)

    assert sum(sizes) == 10
    assert max(sizes) - min(sizes) <= 1


# ============================================================
# Coverage / overlap / count invariants
# ============================================================


def test_complete_index_coverage(dataset: FedMedDataset) -> None:
    """Every global index must appear in exactly one client partition."""

    partitions = partition_dataset(dataset, num_clients=7, seed=42)

    all_indices = sorted(
        index for p in partitions.values() for index in p.indices
    )

    assert all_indices == list(range(len(dataset)))


def test_no_duplicate_indices_within_client(dataset: FedMedDataset) -> None:
    """A single client's indices must not contain duplicates."""

    partitions = partition_dataset(dataset, num_clients=6, seed=42)

    for partition in partitions.values():
        assert len(partition.indices) == len(set(partition.indices))


def test_no_overlap_between_clients(dataset: FedMedDataset) -> None:
    """No global index may be assigned to more than one client."""

    partitions = partition_dataset(dataset, num_clients=6, seed=42)

    index_sets = [set(p.indices) for p in partitions.values()]

    for first in range(len(index_sets)):
        for second in range(first + 1, len(index_sets)):
            assert index_sets[first].isdisjoint(index_sets[second])


def test_total_sample_count_matches_dataset(dataset: FedMedDataset) -> None:
    """The sum of all partition sizes must equal the dataset size."""

    partitions = partition_dataset(dataset, num_clients=9, seed=42)

    total = sum(len(p.indices) for p in partitions.values())

    assert total == len(dataset)


def test_all_clients_non_empty(dataset: FedMedDataset) -> None:
    """Every client must receive at least one sample."""

    partitions = partition_dataset(dataset, num_clients=13, seed=42)

    assert all(len(p.indices) > 0 for p in partitions.values())


# ============================================================
# Determinism / seeding
# ============================================================


def test_deterministic_partitioning_same_seed(dataset: FedMedDataset) -> None:
    """Repeated calls with the same seed must yield identical assignment."""

    first = partition_dataset(dataset, num_clients=5, seed=42)
    second = partition_dataset(dataset, num_clients=5, seed=42)

    for client_id in first:
        assert first[client_id].indices == second[client_id].indices


def test_different_seed_produces_different_partition(dataset: FedMedDataset) -> None:
    """
    Different seeds are expected, in practice, to produce a different
    assignment for a dataset/client-count combination like this one.
    This is a practical sanity check, not a mathematical guarantee.
    """

    first = partition_dataset(dataset, num_clients=5, seed=1)
    second = partition_dataset(dataset, num_clients=5, seed=2)

    assignments_differ = any(
        first[client_id].indices != second[client_id].indices
        for client_id in first
    )

    assert assignments_differ


def test_partition_dataset_does_not_modify_global_random_state(
    dataset: FedMedDataset,
) -> None:
    """
    partition_dataset must not read or consume numpy's global random
    state; it must rely solely on a local np.random.default_rng.
    """

    np.random.seed(123)
    expected_next_value = np.random.rand()

    np.random.seed(123)
    partition_dataset(dataset, num_clients=5, seed=42)
    actual_next_value = np.random.rand()

    assert actual_next_value == expected_next_value


# ============================================================
# Original dataset immutability
# ============================================================


def test_original_dataset_unchanged_after_partition(dataset: FedMedDataset) -> None:
    """Partitioning must not mutate the global dataset in any way."""

    length_before = len(dataset)
    metadata_before = dict(dataset.metadata)
    items_before = {
        index: dataset[index] for index in (0, 1, 25, 50, 74, 99)
    }

    partition_dataset(dataset, num_clients=5, seed=42)

    assert len(dataset) == length_before
    assert dataset.metadata == metadata_before

    for index, item_before in items_before.items():
        assert dataset[index] == item_before


# ============================================================
# ClientPartition metadata
# ============================================================


def test_client_partition_metadata(dataset: FedMedDataset) -> None:
    """Each ClientPartition's metadata must be correct and consistent."""

    partitions = partition_dataset(dataset, num_clients=5, strategy="iid", seed=42)

    for client_id, partition in partitions.items():
        assert partition.metadata["client_id"] == client_id
        assert partition.metadata["num_samples"] == len(partition.indices)
        assert partition.metadata["global_dataset_name"] == dataset.metadata["name"]
        assert partition.metadata["strategy"] == "iid"
        assert partition.metadata["seed"] == 42


# ============================================================
# Invalid input rejection
# ============================================================


def test_zero_clients_rejected(dataset: FedMedDataset) -> None:
    """num_clients=0 must be rejected."""

    with pytest.raises(DataError):
        partition_dataset(dataset, num_clients=0, seed=42)


def test_negative_clients_rejected(dataset: FedMedDataset) -> None:
    """Negative num_clients must be rejected."""

    with pytest.raises(DataError):
        partition_dataset(dataset, num_clients=-3, seed=42)


def test_bool_client_count_rejected(dataset: FedMedDataset) -> None:
    """Boolean num_clients must be rejected despite bool being an int subclass."""

    with pytest.raises(DataError):
        partition_dataset(dataset, num_clients=True, seed=42)

    with pytest.raises(DataError):
        partition_dataset(dataset, num_clients=False, seed=42)


def test_more_clients_than_samples_rejected(small_dataset: FedMedDataset) -> None:
    """num_clients exceeding the dataset size must be rejected."""

    with pytest.raises(DataError):
        partition_dataset(small_dataset, num_clients=11, seed=42)


def test_invalid_dataset_type_rejected() -> None:
    """A non-FedMedDataset input must be rejected."""

    with pytest.raises(DataError):
        partition_dataset([1, 2, 3], num_clients=1, seed=42)  # type: ignore[arg-type]


def test_invalid_strategy_rejected(dataset: FedMedDataset) -> None:
    """An unsupported strategy name must be rejected."""

    with pytest.raises(DataError):
        partition_dataset(dataset, num_clients=5, strategy="dirichlet", seed=42)


# ============================================================
# Partition dataset view retrieval (via partition_dataset)
# ============================================================


def test_partition_dataset_retrieval_matches_global(dataset: FedMedDataset) -> None:
    """partition.dataset[i] must match global_dataset[partition.indices[i]]."""

    partitions = partition_dataset(dataset, num_clients=5, seed=42)

    for partition in partitions.values():
        for local_index, global_index in enumerate(partition.indices):
            assert partition.dataset[local_index] == dataset[global_index]


# ============================================================
# PartitionView index contract
# ============================================================


def test_partition_view_positive_index(partition_view: PartitionView) -> None:
    """A valid positive local index resolves to the correct global item."""

    assert partition_view[0] == partition_view._dataset[5]
    assert partition_view[2] == partition_view._dataset[15]


def test_partition_view_negative_index(partition_view: PartitionView) -> None:
    """A valid negative local index follows standard Python semantics."""

    assert partition_view[-1] == partition_view._dataset[15]
    assert partition_view[-3] == partition_view._dataset[5]


def test_partition_view_numpy_integer_index(partition_view: PartitionView) -> None:
    """NumPy integer indices (e.g. np.int64) must be accepted."""

    assert partition_view[np.int64(1)] == partition_view._dataset[10]


def test_partition_view_string_index_rejected(partition_view: PartitionView) -> None:
    """A string index must be rejected."""

    with pytest.raises(DataError):
        partition_view["0"]


def test_partition_view_bool_index_rejected(partition_view: PartitionView) -> None:
    """A boolean index must be rejected despite bool being an int subclass."""

    with pytest.raises(DataError):
        partition_view[True]

    with pytest.raises(DataError):
        partition_view[False]


def test_partition_view_out_of_range_positive_index_rejected(
    partition_view: PartitionView,
) -> None:
    """An out-of-range positive local index must be rejected."""

    with pytest.raises(DataError):
        partition_view[3]


def test_partition_view_out_of_range_negative_index_rejected(
    partition_view: PartitionView,
) -> None:
    """An out-of-range negative local index must be rejected."""

    with pytest.raises(DataError):
        partition_view[-4]