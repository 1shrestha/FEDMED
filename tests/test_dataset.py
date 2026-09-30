"""
Tests for the FedMed dataset foundation.

This module tests:

- FedMedDataset construction
- Length
- Item retrieval
- Deterministic retrieval
- Metadata
- Empty dataset rejection
- None input rejection
- Sample/target mismatch rejection
- Invalid container rejection
- Invalid indexing rejection
- DataError behavior
- Mutation-after-construction isolation
"""

import numpy as np
import pytest
import torch

from src.common.exceptions import DataError
from src.data.dataset import FedMedDataset


@pytest.fixture
def samples() -> list[int]:
    """A small deterministic sample set."""

    return [10, 20, 30, 40]


@pytest.fixture
def targets() -> list[int]:
    """Targets matching the samples fixture."""

    return [0, 1, 0, 1]


@pytest.fixture
def dataset(samples: list[int], targets: list[int]) -> FedMedDataset:
    """A valid, freshly constructed dataset."""

    return FedMedDataset(
        samples=samples,
        targets=targets,
        name="phase1_dataset",
    )


# ============================================================
# Construction / basic behavior
# ============================================================


def test_dataset_constructs_successfully(dataset: FedMedDataset) -> None:
    """Verify a valid dataset can be constructed."""

    assert len(dataset) == 4


def test_dataset_length(dataset: FedMedDataset) -> None:
    """Verify __len__ reflects the number of samples."""

    assert len(dataset) == 4


def test_dataset_item_retrieval(dataset: FedMedDataset) -> None:
    """Verify __getitem__ returns the correct (sample, target) pair."""

    sample, target = dataset[0]

    assert sample == 10
    assert target == 0

    sample, target = dataset[3]

    assert sample == 40
    assert target == 1


def test_dataset_negative_index(dataset: FedMedDataset) -> None:
    """Verify negative indices follow standard Python semantics."""

    sample, target = dataset[-1]

    assert sample == 40
    assert target == 1


def test_dataset_deterministic_retrieval(dataset: FedMedDataset) -> None:
    """Verify repeated access to the same index is identical."""

    first = dataset[2]
    second = dataset[2]

    assert first == second


def test_dataset_metadata(dataset: FedMedDataset) -> None:
    """Verify metadata contains expected, non-sensitive fields."""

    metadata = dataset.metadata

    assert metadata["name"] == "phase1_dataset"
    assert metadata["size"] == 4
    assert metadata["sample_container_type"] == "tuple"
    assert metadata["target_container_type"] == "tuple"


def test_dataset_repr(dataset: FedMedDataset) -> None:
    """Verify __repr__ exposes only non-sensitive summary info."""

    text = repr(dataset)

    assert "phase1_dataset" in text
    assert "4" in text


def test_dataset_supports_tuples(targets: list[int]) -> None:
    """Verify the dataset accepts tuples, not only lists."""

    tuple_samples = (1, 2, 3)
    tuple_targets = (0, 1, 0)

    ds = FedMedDataset(
        samples=tuple_samples,
        targets=tuple_targets,
        name="tuple_dataset",
    )

    assert len(ds) == 3
    assert ds[1] == (2, 1)


def test_dataset_generic_target_types() -> None:
    """Verify targets are not constrained to int (e.g. float regression)."""

    ds = FedMedDataset(
        samples=[[0.1, 0.2], [0.3, 0.4]],
        targets=[3.14, -1.5],
        name="regression_dataset",
    )

    sample, target = ds[0]

    assert sample == [0.1, 0.2]
    assert target == 3.14


# ============================================================
# Container type compatibility (numpy / torch / generic)
# ============================================================


def test_dataset_supports_numpy_array_samples() -> None:
    """Verify a numpy.ndarray is accepted as a samples container."""

    samples = np.array([[1, 2], [3, 4], [5, 6]])
    targets = [0, 1, 0]

    ds = FedMedDataset(samples=samples, targets=targets, name="numpy_dataset")

    assert len(ds) == 3

    sample, target = ds[1]

    assert np.array_equal(sample, np.array([3, 4]))
    assert target == 1
    assert ds.metadata["sample_container_type"] == "ndarray"


def test_dataset_supports_torch_tensor_samples() -> None:
    """Verify a torch.Tensor is accepted as a samples container."""

    samples = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.tensor([0, 1])

    ds = FedMedDataset(samples=samples, targets=targets, name="tensor_dataset")

    assert len(ds) == 2

    sample, target = ds[0]

    assert torch.equal(sample, torch.tensor([1.0, 2.0]))
    assert target == torch.tensor(0)
    assert ds.metadata["sample_container_type"] == "Tensor"


def test_dataset_supports_mixed_container_types() -> None:
    """Verify samples and targets may use different container types."""

    samples = np.array([1, 2, 3])
    targets = [0, 1, 0]

    ds = FedMedDataset(samples=samples, targets=targets, name="mixed_dataset")

    assert len(ds) == 3
    assert ds.metadata["sample_container_type"] == "ndarray"
    assert ds.metadata["target_container_type"] == "tuple"


def test_dataset_supports_generic_custom_container() -> None:
    """
    Verify any object satisfying the SizedIndexable protocol (len +
    integer indexing) is accepted, not only built-in/numpy/torch types.
    """

    class CustomContainer:
        def __init__(self, data: list[int]) -> None:
            self._data = data

        def __len__(self) -> int:
            return len(self._data)

        def __getitem__(self, index: int):
            return self._data[index]

    samples = CustomContainer([1, 2, 3])
    targets = CustomContainer([0, 1, 0])

    ds = FedMedDataset(samples=samples, targets=targets, name="custom_dataset")

    assert len(ds) == 3
    assert ds[1] == (2, 1)


def test_numpy_integer_index_accepted(dataset: FedMedDataset) -> None:
    """
    Verify numpy integer types (e.g. from a future partitioner using
    numpy.random/argsort) are accepted as valid indices.
    """

    sample, target = dataset[np.int64(2)]

    assert sample == 30
    assert target == 0


# ============================================================
# Mutation isolation
# ============================================================


def test_dataset_isolated_from_external_list_mutation(
    samples: list[int], targets: list[int]
) -> None:
    """
    Verify that mutating the caller's original list after construction
    does not affect the dataset (list -> tuple snapshot ownership
    contract).
    """

    ds = FedMedDataset(samples=samples, targets=targets, name="mutation_test")

    samples.append(999)
    targets.append(1)
    samples.clear()

    assert len(ds) == 4
    assert ds[0] == (10, 0)


def test_dataset_holds_ndarray_by_reference() -> None:
    """
    Verify numpy.ndarray containers are held by reference (not
    copied), per the type-aware ownership model: this is a deliberate
    memory/performance trade-off, not a claim that arrays are
    immutable. Element-level in-place mutation of the original array
    remains visible through the dataset (documented, expected
    behavior — not a bug).
    """

    samples = np.array([1, 2, 3])
    targets = np.array([0, 1, 0])

    ds = FedMedDataset(samples=samples, targets=targets, name="ref_test")

    samples[0] = 999  # in-place element mutation

    sample, _ = ds[0]

    assert sample == 999  # visible through the dataset, as documented


def test_dataset_holds_tuple_by_reference(
    samples: list[int], targets: list[int]
) -> None:
    """Verify tuple input is stored directly, without an extra copy."""

    tuple_samples = tuple(samples)
    tuple_targets = tuple(targets)

    ds = FedMedDataset(
        samples=tuple_samples, targets=tuple_targets, name="tuple_ref_test"
    )

    assert ds.metadata["sample_container_type"] == "tuple"


# ============================================================
# Validation failures
# ============================================================


def test_none_samples_raises_error(targets: list[int]) -> None:
    """Verify None samples raise DataError."""

    with pytest.raises(DataError):
        FedMedDataset(samples=None, targets=targets, name="invalid")


def test_none_targets_raises_error(samples: list[int]) -> None:
    """Verify None targets raise DataError."""

    with pytest.raises(DataError):
        FedMedDataset(samples=samples, targets=None, name="invalid")


def test_empty_dataset_raises_error() -> None:
    """Verify an empty dataset is rejected."""

    with pytest.raises(DataError):
        FedMedDataset(samples=[], targets=[], name="empty")


def test_length_mismatch_raises_error() -> None:
    """Verify mismatched samples/targets lengths raise DataError."""

    with pytest.raises(DataError):
        FedMedDataset(samples=[1, 2, 3], targets=[0, 1], name="mismatch")


def test_invalid_sample_container_raises_error(targets: list[int]) -> None:
    """Verify a non-sequence samples container raises DataError."""

    with pytest.raises(DataError):
        FedMedDataset(samples=42, targets=targets, name="invalid")


def test_invalid_target_container_raises_error(samples: list[int]) -> None:
    """Verify a non-sequence targets container raises DataError."""

    with pytest.raises(DataError):
        FedMedDataset(samples=samples, targets=object(), name="invalid")


def test_string_samples_rejected(targets: list[int]) -> None:
    """
    Verify a bare string is rejected as a samples container, even
    though strings support len() and indexing (this would silently
    iterate characters, not samples).
    """

    with pytest.raises(DataError):
        FedMedDataset(samples="abcd", targets=[0, 1, 0, 1], name="invalid")


def test_bytes_samples_rejected(targets: list[int]) -> None:
    """Verify a bytes object is rejected as a samples container."""

    with pytest.raises(DataError):
        FedMedDataset(samples=b"abcd", targets=[0, 1, 0, 1], name="invalid")


def test_dict_samples_rejected(targets: list[int]) -> None:
    """
    Verify a dict is rejected as a samples container, even though it
    has len() and __getitem__ (positional indexing would be
    ambiguous/incorrect: d[0] means "look up key 0", not "position 0").
    """

    with pytest.raises(DataError):
        FedMedDataset(
            samples={0: "a", 1: "b", 2: "c", 3: "d"},
            targets=[0, 1, 0, 1],
            name="invalid",
        )


def test_container_without_getitem_rejected(targets: list[int]) -> None:
    """Verify a sized-but-not-indexable container (e.g. a set) is rejected."""

    with pytest.raises(DataError):
        FedMedDataset(samples={1, 2, 3, 4}, targets=[0, 1, 0, 1], name="invalid")


# ============================================================
# Malformed len() / unsized containers (hardening pass)
# ============================================================


class _BrokenLenContainer:
    """
    Structurally satisfies SizedIndexable (has __len__ and
    __getitem__) but __len__ raises when actually called. Used to
    verify malformed containers cannot leak a raw exception out of
    dataset construction.
    """

    def __len__(self) -> int:
        raise RuntimeError("simulated broken len()")

    def __getitem__(self, index: int):
        return index


def test_broken_len_samples_raises_dataerror(targets: list[int]) -> None:
    """
    Verify a samples container whose __len__ raises produces a
    DataError, not a raw RuntimeError leaking out of construction.
    """

    with pytest.raises(DataError):
        FedMedDataset(
            samples=_BrokenLenContainer(),
            targets=targets,
            name="broken_len",
        )


def test_broken_len_targets_raises_dataerror(samples: list[int]) -> None:
    """Verify the same guard applies to the targets container."""

    with pytest.raises(DataError):
        FedMedDataset(
            samples=samples,
            targets=_BrokenLenContainer(),
            name="broken_len",
        )


def test_zero_dimensional_numpy_array_rejected(targets: list[int]) -> None:
    """
    Verify a zero-dimensional numpy array (a scalar, not a sized
    container) is rejected with DataError rather than leaking numpy's
    raw "len() of unsized object" TypeError.
    """

    with pytest.raises(DataError):
        FedMedDataset(samples=np.array(5), targets=targets, name="scalar_array")


def test_zero_dimensional_torch_tensor_rejected(targets: list[int]) -> None:
    """
    Verify a zero-dimensional torch tensor (a scalar) is rejected
    with DataError rather than leaking torch's raw "len() of a 0-d
    tensor" TypeError.
    """

    with pytest.raises(DataError):
        FedMedDataset(
            samples=torch.tensor(5), targets=targets, name="scalar_tensor"
        )


def test_zero_dimensional_numpy_array_as_targets_rejected(
    samples: list[int],
) -> None:
    """Verify the same zero-dimensional guard applies to targets."""

    with pytest.raises(DataError):
        FedMedDataset(samples=samples, targets=np.array(5), name="scalar_target")


class _MutableContainer:
    """
    A reference-held custom SizedIndexable container whose length can
    change after construction (backed by a mutable list the test can
    resize directly). Used to verify FedMedDataset detects structural
    mutation of reference-held containers without relying on
    numpy/torch resize semantics.
    """

    def __init__(self, data: list[int]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]


def test_getitem_detects_samples_container_resized_after_construction(
    targets: list[int],
) -> None:
    """
    Verify that if a reference-held samples container is resized after
    construction, __getitem__ raises DataError instead of returning
    misaligned or stale data.
    """

    container = _MutableContainer([10, 20, 30, 40])

    ds = FedMedDataset(samples=container, targets=targets, name="resized_samples")

    assert ds[0] == (10, 0)  # works before mutation

    container.data.append(50)  # resize the referenced container

    with pytest.raises(DataError):
        _ = ds[0]


def test_getitem_detects_targets_container_resized_after_construction(
    samples: list[int],
) -> None:
    """
    Verify that if a reference-held targets container is resized after
    construction, __getitem__ raises DataError instead of returning
    misaligned or stale data.
    """

    container = _MutableContainer([0, 1, 0, 1])

    ds = FedMedDataset(samples=samples, targets=container, name="resized_targets")

    assert ds[0] == (10, 0)  # works before mutation

    container.data.pop()  # resize the referenced container

    with pytest.raises(DataError):
        _ = ds[0]


def test_empty_name_raises_error(samples: list[int], targets: list[int]) -> None:
    """Verify an empty or whitespace-only name raises DataError."""

    with pytest.raises(DataError):
        FedMedDataset(samples=samples, targets=targets, name="   ")


def test_invalid_index_type_raises_error(dataset: FedMedDataset) -> None:
    """Verify a non-integer index raises DataError."""

    with pytest.raises(DataError):
        _ = dataset["0"]


def test_boolean_index_raises_error(dataset: FedMedDataset) -> None:
    """Verify a boolean index is rejected despite bool being an int subclass."""

    with pytest.raises(DataError):
        _ = dataset[True]


def test_out_of_range_positive_index_raises_error(dataset: FedMedDataset) -> None:
    """Verify an index beyond the dataset size raises DataError."""

    with pytest.raises(DataError):
        _ = dataset[100]


def test_out_of_range_negative_index_raises_error(dataset: FedMedDataset) -> None:
    """Verify an excessively negative index raises DataError."""

    with pytest.raises(DataError):
        _ = dataset[-100]