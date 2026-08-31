"""
Dataset foundation for FedMed.

This module defines FedMedDataset, a validated, deterministic,
framework-compatible container of (sample, target) pairs.

FedMedDataset represents DATA only. It intentionally does NOT know
about hospitals, clients, federated partitioning, training, models,
device placement, or Flower. Those responsibilities live in later
Phase 1 modules (partitioner.py, loader.py, trainer.py) and beyond.

This module intentionally does NOT contain:

- Federated partitioning logic
- DataLoader / batching / sampling
- Transforms, normalization, or augmentation
- Training loops, losses, or optimizers
- GPU/device placement
- Flower-specific code
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from torch.utils.data import Dataset

from src.common.exceptions import DataError


@runtime_checkable
class SizedIndexable(Protocol):
    """
    Structural contract for anything FedMedDataset can wrap.

    Any object supporting ``len()`` and integer ``[]`` indexing
    satisfies this protocol — including ``list``, ``tuple``,
    ``numpy.ndarray``, and ``torch.Tensor``, without hardcoding a
    whitelist of accepted concrete types. This is intentionally
    structural (duck-typed) rather than nominal: FedMed should not
    reject a valid container merely because it doesn't register as
    ``typing.Sequence`` (numpy arrays and torch tensors do not).
    """

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...


class FedMedDataset(Dataset):
    """
    In-memory dataset of (sample, target) pairs.

    This is the Phase 1 foundation dataset: a simple, generic,
    testable container. It does not assume a specific task
    (classification, segmentation, regression), a specific sample
    type (list element, numpy array, torch tensor, ...), or that all
    future FedMed data will fit in memory — it is simply the smallest
    correct abstraction for the current phase.

    Ownership:
        Ownership handling depends on the input container type:

        - ``list`` inputs are snapshotted into a ``tuple`` at
          construction time. This is a shallow operation (element
          references are shared, not copied) that protects the
          dataset from later mutation of the caller's original list
          (e.g. ``append``/``clear``/``sort``).
        - ``tuple``, ``numpy.ndarray``, ``torch.Tensor``, and any
          other object satisfying :class:`SizedIndexable` are stored
          by direct reference, not defensively copied. This is a
          deliberate memory/performance trade-off for Phase 1:
          array/tensor objects may be large, and copying them on
          every dataset construction would be wasteful. This is
          *not* because these types are inherently immutable or
          fixed-size — a ``tuple`` is immutable, but a
          ``numpy.ndarray``/``torch.Tensor`` is not, and neither this
          class nor Python itself prevents the caller from mutating
          such an object in place after construction. The practical
          consequence: if the caller mutates elements of an
          externally-owned array/tensor in place after passing it in
          (e.g. ``samples[0] = ...``), that change *is* visible
          through the dataset, since no independent copy exists.
          Callers that need isolation from an array/tensor they
          continue to hold onto are responsible for copying it
          themselves before construction (e.g. ``array.copy()``).

        In all cases, mutating an individual element *in place*
        after construction (e.g. ``samples[0] += 1`` on a numpy
        array, or mutating a mutable object stored inside a tuple)
        remains visible through the dataset. Only external mutation
        of the ``list`` container itself (its length/order) is
        guarded against; reference-held containers are not — but
        ``__getitem__`` does detect if a reference-held container's
        *length* has changed since construction and raises
        ``DataError`` rather than returning misaligned data.

    Determinism:
        ``__getitem__`` is a pure lookup with no randomness. Sampling,
        shuffling, and augmentation belong in later components
        (DataLoader, samplers, partitioner), not here.

    Naming:
        ``name`` is expected to be a short, non-sensitive technical
        identifier (e.g. ``"phase1_dataset"``, ``"hospital_a_train"``).
        It is surfaced verbatim in :attr:`metadata`, ``repr()``, and
        in every ``DataError`` message raised by this class, so it
        must never contain patient identifiers or other sensitive
        information.
    """

    def __init__(
        self,
        samples: SizedIndexable | None,
        targets: SizedIndexable | None,
        name: str,
    ) -> None:
        """
        Construct a validated FedMedDataset.

        Args:
            samples: Indexable, sized container of input samples
                (list, tuple, numpy.ndarray, torch.Tensor, or any
                object supporting ``len()`` and integer indexing).
            targets: Indexable, sized container of targets, one per
                sample. No type is assumed — targets may be class
                indices, label vectors, segmentation masks, or
                regression values.
            name: Human-readable identifier for this dataset, used
                verbatim in metadata, repr(), and error messages.
                Must be a short, non-sensitive technical identifier —
                see the "Naming" note in the class docstring.

        Raises:
            DataError: If any input fails validation (see class-level
                validation rules).
        """

        self._validate_name(name)
        self._validate_container(samples, "samples")
        self._validate_container(targets, "targets")

        normalized_samples = self._normalize_container(samples)
        normalized_targets = self._normalize_container(targets)

        samples_length = self._safe_len(normalized_samples, "samples")
        targets_length = self._safe_len(normalized_targets, "targets")

        if samples_length != targets_length:
            raise DataError(
                f"Dataset '{name}': samples and targets length "
                f"mismatch (samples={samples_length}, "
                f"targets={targets_length})."
            )

        if samples_length == 0:
            raise DataError(
                f"Dataset '{name}': cannot construct an empty dataset."
            )

        self._name = name
        self._samples = normalized_samples
        self._targets = normalized_targets
        self._length = samples_length

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise DataError(
                "Dataset name must be a non-empty string."
            )

    @staticmethod
    def _validate_container(value: Any, label: str) -> None:
        if value is None:
            raise DataError(f"Dataset '{label}' cannot be None.")

        if isinstance(value, (str, bytes)):
            raise DataError(
                f"Dataset '{label}' cannot be a str/bytes object "
                f"(would iterate characters/bytes, not samples)."
            )

        if isinstance(value, Mapping):
            raise DataError(
                f"Dataset '{label}' cannot be a mapping (dict-like "
                f"objects have ambiguous positional indexing)."
            )

        if not isinstance(value, SizedIndexable):
            raise DataError(
                f"Dataset '{label}' must support len() and integer "
                f"indexing (e.g. list, tuple, numpy.ndarray, "
                f"torch.Tensor), got {type(value).__name__}."
            )

    @staticmethod
    def _normalize_container(value: SizedIndexable) -> SizedIndexable:
        """
        Apply the type-aware ownership strategy described in the
        class docstring: snapshot ``list`` inputs into a ``tuple``;
        hold every other container (``tuple``, ``numpy.ndarray``,
        ``torch.Tensor``, or any other :class:`SizedIndexable`) by
        direct reference without copying.
        """

        if isinstance(value, list):
            return tuple(value)

        return value

    @staticmethod
    def _safe_len(value: SizedIndexable, label: str) -> int:
        """
        Compute ``len(value)`` without letting a malformed or
        incompatible container leak a raw ``TypeError``/``ValueError``/
        ``RuntimeError`` out of dataset construction.

        Satisfying :class:`SizedIndexable` structurally (having a
        ``__len__`` attribute) does not guarantee that calling it
        succeeds or behaves sensibly. Two concrete cases this guards
        against:

        - A zero-dimensional ``numpy.ndarray`` (``np.array(5)``) or a
          scalar ``torch.Tensor`` (``torch.tensor(5)``) both define
          ``__len__`` but raise ``TypeError`` when it is called,
          since a scalar has no length.
        - A custom container whose ``__len__`` implementation itself
          raises (a bug in that container, or a container that is
          only partially initialized).

        In both cases this should surface as a domain-level
        ``DataError`` describing the problem, not an unhandled
        framework exception from deep inside dataset construction.

        Args:
            value: The (already normalized) container to measure.
            label: "samples" or "targets", used in the error message.

        Returns:
            The container's length.

        Raises:
            DataError: If ``len(value)`` raises for any reason.
        """

        try:
            return len(value)
        except Exception as exc:
            raise DataError(
                f"Dataset '{label}' does not support a valid len() "
                f"(e.g. a zero-dimensional array/tensor, or a "
                f"malformed container): {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self._length

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Return the (sample, target) pair at the given index.

        Args:
            index: Integer-valued index. Accepts native Python ``int``
                as well as NumPy integer types (e.g. ``numpy.int64``),
                since a future partitioner producing indices via
                ``numpy.random``/``numpy.argsort`` will naturally
                yield NumPy integers rather than plain ``int``.
                Negative indices are supported with standard Python
                semantics (e.g. -1 is the last item).

        Returns:
            A (sample, target) tuple.

        Raises:
            DataError: If index is not an integer type or is out of
                range, or if a referenced (non-list-derived) samples
                or targets container has structurally changed size
                since construction (see class docstring "Ownership"
                note — reference-held containers are not defensively
                copied, so an externally-owned array/tensor/custom
                container that is resized after construction can
                drift out of sync with the dataset's cached length;
                this is detected here rather than silently returning
                stale/misaligned data).
        """

        if isinstance(index, bool) or not isinstance(index, numbers.Integral):
            raise DataError(
                f"Dataset '{self._name}': index must be an integer, "
                f"got {type(index).__name__}."
            )

        index = int(index)

        if index < -self._length or index >= self._length:
            raise DataError(
                f"Dataset '{self._name}': index {index} out of range "
                f"for dataset of size {self._length}."
            )

        self._check_container_unchanged(self._samples, "samples")
        self._check_container_unchanged(self._targets, "targets")

        return self._samples[index], self._targets[index]

    def _check_container_unchanged(self, value: SizedIndexable, label: str) -> None:
        """
        Verify a reference-held container still has the length it had
        at construction time.

        Only ``list`` inputs are snapshotted into an immutable
        ``tuple``; every other accepted container (``tuple``,
        ``numpy.ndarray``, ``torch.Tensor``, custom
        :class:`SizedIndexable` objects) is stored by direct
        reference without copying, per the documented ownership
        strategy. If the caller resizes such a container after
        construction, the dataset's cached ``_length`` would
        otherwise silently disagree with the live container, risking
        misaligned sample/target pairs or an out-of-range index deep
        inside a downstream training loop. This check surfaces that
        as an immediate, clear ``DataError`` instead.

        Args:
            value: The stored samples or targets container.
            label: "samples" or "targets", used in the error message.

        Raises:
            DataError: If the container's current length differs from
                the length recorded at construction.
        """

        current_length = self._safe_len(value, label)

        if current_length != self._length:
            raise DataError(
                f"Dataset '{self._name}': '{label}' container changed "
                f"size after construction (was {self._length}, now "
                f"{current_length}). Reference-held containers "
                f"(anything other than an original list input) are "
                f"not defensively copied; resizing one after "
                f"constructing the dataset is not supported."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self._name!r}, "
            f"size={self._length})"
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Return non-sensitive metadata about this dataset.

        Deliberately excludes any per-sample content, patient
        identifiers, or other sensitive information — only shape-level
        and container-type facts about the dataset as a whole. The
        container type names (e.g. "tuple", "ndarray", "Tensor") are
        included because they are directly useful for future
        loader.py/trainer.py integration when deciding how to batch
        or convert samples.
        """

        return {
            "name": self._name,
            "size": self._length,
            "sample_container_type": type(self._samples).__name__,
            "target_container_type": type(self._targets).__name__,
        }