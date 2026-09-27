"""
Federation checkpoint persistence for FedMed.

This module provides the framework-independent durability boundary
for a FedMed federation session.

Responsibilities
----------------
This module is responsible for:

- describing a resumable federation checkpoint
- recording the structural parameter contract alongside the values
- validating a checkpoint against that recorded contract
- persisting checkpoints atomically to the local filesystem
- discovering the most recent persisted checkpoint

This module intentionally does NOT:

- implement aggregation
- implement client selection
- implement local training or evaluation
- orchestrate federated rounds
- own federation progress or global state
- import Flower or any runtime/transport API
- implement remote or distributed storage

The checkpoint is a *description* of server-owned state. Deciding
*when* to persist is the caller's decision, and restoring state into
a live server is ``FederatedServer``'s decision. This module only
moves validated state to and from disk.

Architecture
------------

    FederatedServer
           |
           | global parameters + completed round
           v
    FederationCheckpoint
           |
           v
    FederationCheckpointStore
           |
           v
    filesystem (atomic replace)

For example:

    store = FederationCheckpointStore.from_config(checkpoint_config)
    store.save(checkpoint)
    restored = store.load_latest()

Storage format
--------------
Checkpoints are stored as a single uncompressed ``.npz`` archive
containing:

- ``round``          int64 scalar, the completed round number
- ``contract_names``  unicode array, state_dict keys in order
- ``contract_shapes`` int64 array, flattened shapes in order
- ``contract_dtypes`` unicode array, dtype strings in order
- ``param_<i>``       the i-th parameter value

The recorded contract makes a checkpoint self-describing, so a
mismatch between the saved federation and the current one is
detected instead of silently producing a corrupt resume.

Crash safety
------------
``save`` writes to a temporary file in the destination directory
and then atomically replaces the target with ``os.replace``. A crash
during serialization therefore cannot damage an existing
checkpoint: the previous file remains intact and the temporary file
is left behind as an ignorable artifact.
"""

from __future__ import annotations

import os
import tempfile

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.common.exceptions import FederatedLearningError
from src.fl.parameters import (
    ParameterContract,
    ParameterPayload,
    ParameterSpec,
    copy_parameters,
    validate_parameters,
)


# ============================================================
# Exceptions
# ============================================================


class CheckpointError(FederatedLearningError):
    """
    Raised when a federation checkpoint cannot be created,
    persisted, read, or validated.

    This is intentionally more specific than the generic
    FederatedLearningError while remaining inside the existing
    FedMed federated exception hierarchy.
    """

    pass


# ============================================================
# Checkpoint
# ============================================================


@dataclass(frozen=True)
class FederationCheckpoint:
    """
    Immutable, self-describing snapshot of resumable server state.

    Attributes:
        completed_round:
            One-based number of the most recently completed
            federated round. Zero means no round has completed.

        contract:
            Structural parameter layout this checkpoint was
            captured against.

        parameters:
            Global parameter payload as of ``completed_round``.

    Notes:
    The parameter payload is defensively copied on construction,
    so a checkpoint cannot be mutated through the arrays it was
    built from, nor after it has been handed to a store.
    """

    completed_round: int
    contract: ParameterContract
    parameters: ParameterPayload

    def __post_init__(self) -> None:
        """Validate the round number, contract, and payload."""

        if isinstance(self.completed_round, bool) or not isinstance(
            self.completed_round, (int, np.integer)
        ):
            raise CheckpointError(
                "FederationCheckpoint.completed_round must be an "
                f"integer, got {type(self.completed_round).__name__}."
            )

        if self.completed_round < 0:
            raise CheckpointError(
                "FederationCheckpoint.completed_round must be >= 0, "
                f"got {self.completed_round}."
            )

        if not isinstance(self.contract, ParameterContract):
            raise CheckpointError(
                "FederationCheckpoint.contract must be a "
                f"ParameterContract, got {type(self.contract).__name__}."
            )

        if not isinstance(self.parameters, (list, tuple)):
            raise CheckpointError(
                "FederationCheckpoint.parameters must be a list or "
                f"tuple of NumPy arrays, got "
                f"{type(self.parameters).__name__}."
            )

        if not self.parameters:
            raise CheckpointError(
                "FederationCheckpoint.parameters must not be empty."
            )

        try:
            validate_parameters(self.parameters, self.contract)

        except FederatedLearningError as exc:
            raise CheckpointError(
                "FederationCheckpoint parameters do not satisfy "
                f"their own recorded contract: {exc}"
            ) from exc

        object.__setattr__(
            self,
            "parameters",
            copy_parameters(self.parameters),
        )

    @property
    def parameter_count(self) -> int:
        """Return the number of parameters in the payload."""

        return len(self.parameters)

    @classmethod
    def from_parameters(
        cls,
        completed_round: int,
        parameters: Sequence[np.ndarray],
        contract: ParameterContract,
    ) -> "FederationCheckpoint":
        """
        Build a checkpoint from a raw parameter payload.

        This is the intended construction path for callers that
        hold a payload and a contract rather than a checkpoint.

        Args:
            completed_round:
                One-based completed round number.

            parameters:
                Global parameter payload to capture.

            contract:
                Structural layout the payload must satisfy.

        Returns:
            A validated FederationCheckpoint.

        Raises:
            CheckpointError:
                If the round number, contract, or payload is
                invalid.
        """

        return cls(
            completed_round=completed_round,
            contract=contract,
            parameters=list(parameters),
        )

    def validate_against(
        self,
        contract: ParameterContract,
    ) -> None:
        """
        Verify this checkpoint belongs to a given contract.

        This detects a checkpoint captured against a different model
        before its values are loaded into a live federation.

        Args:
            contract:
                Expected parameter layout of the target federation.

        Raises:
            CheckpointError:
                If the recorded and expected contracts differ.
        """

        if not isinstance(contract, ParameterContract):
            raise CheckpointError(
                "Expected contract must be a ParameterContract, got "
                f"{type(contract).__name__}."
            )

        if self.contract.count != contract.count:
            raise CheckpointError(
                "Checkpoint parameter count mismatch: checkpoint has "
                f"{self.contract.count}, expected {contract.count}."
            )

        for index, (saved, expected) in enumerate(
            zip(self.contract.specs, contract.specs)
        ):
            if saved.name != expected.name:
                raise CheckpointError(
                    "Checkpoint parameter name mismatch at index "
                    f"{index}: checkpoint='{saved.name}', "
                    f"expected='{expected.name}'."
                )

            if saved.shape != expected.shape:
                raise CheckpointError(
                    f"Checkpoint shape mismatch for '{saved.name}': "
                    f"checkpoint={saved.shape}, "
                    f"expected={expected.shape}."
                )

            if saved.dtype != expected.dtype:
                raise CheckpointError(
                    f"Checkpoint dtype mismatch for '{saved.name}': "
                    f"checkpoint={saved.dtype}, "
                    f"expected={expected.dtype}."
                )

    def restore_parameters(self) -> ParameterPayload:
        """
        Return an independent copy of the checkpoint payload.

        Returns:
            A new list of NumPy-array copies safe to hand to a
            live server.
        """

        return copy_parameters(self.parameters)


# ============================================================
# Store
# ============================================================


class FederationCheckpointStore:
    """
    Filesystem-backed store for federation checkpoints.

    The store owns checkpoint persistence only. It does not know
    about federated rounds, strategies, or aggregators.

    Parameters:
        directory:
            Destination directory for checkpoint files. It is
            created if missing.

        save_every_round:
            Minimum number of rounds between automatic saves.
            Must be a positive integer.

    Notes:
    The store is safe to construct for an empty directory. A
    missing checkpoint is reported as ``None`` rather than as an
    error, because "nothing saved yet" is a normal state.
    """

    FILENAME_PREFIX = "round_"
    FILENAME_SUFFIX = ".npz"

    def __init__(
        self,
        directory: str | Path,
        *,
        save_every_round: int = 1,
    ) -> None:
        if not isinstance(directory, (str, Path)):
            raise CheckpointError(
                "Checkpoint directory must be a string or Path, got "
                f"{type(directory).__name__}."
            )

        if isinstance(directory, str) and not directory.strip():
            raise CheckpointError(
                "Checkpoint directory must be a non-empty path."
            )

        if (
            isinstance(save_every_round, bool)
            or not isinstance(save_every_round, int)
        ):
            raise CheckpointError(
                "save_every_round must be an integer, got "
                f"{type(save_every_round).__name__}."
            )

        if save_every_round < 1:
            raise CheckpointError(
                "save_every_round must be >= 1, got "
                f"{save_every_round}."
            )

        self._directory = Path(directory)
        self._save_every_round = save_every_round

        try:
            self._directory.mkdir(parents=True, exist_ok=True)

        except OSError as exc:
            raise CheckpointError(
                f"Failed to create checkpoint directory "
                f"{self._directory}: {exc}"
            ) from exc

    # --------------------------------------------------------
    # Construction helpers
    # --------------------------------------------------------

    @classmethod
    def from_config(cls, config) -> "FederationCheckpointStore":
        """
        Build a store from a ``CheckpointConfig``.

        Args:
            config:
                FedMed ``CheckpointConfig`` exposing ``directory``
                and ``save_every_round``.

        Returns:
            A configured FederationCheckpointStore.

        Raises:
            CheckpointError:
                If the config does not provide the required
                attributes or is otherwise invalid.
        """

        directory = getattr(config, "directory", None)
        save_every_round = getattr(config, "save_every_round", None)

        if directory is None or save_every_round is None:
            raise CheckpointError(
                "from_config() requires a CheckpointConfig exposing "
                "'directory' and 'save_every_round', got "
                f"{type(config).__name__}."
            )

        return cls(
            directory,
            save_every_round=save_every_round,
        )

    # --------------------------------------------------------
    # Properties
    # --------------------------------------------------------

    @property
    def directory(self) -> Path:
        """Return the checkpoint destination directory."""

        return self._directory

    @property
    def save_every_round(self) -> int:
        """Return the configured save interval."""

        return self._save_every_round

    def should_save(self, completed_round: int) -> bool:
        """
        Return whether a round qualifies for an automatic save.

        A round qualifies when it is a non-negative integer that
        falls on the configured interval boundary. Round zero is
        eligible so that the initial global model can be captured
        before the first round runs.

        Args:
            completed_round:
                One-based completed round number.

        Returns:
            True if the round should be persisted.
        """

        if isinstance(completed_round, bool) or not isinstance(
            completed_round, int
        ):
            return False

        if completed_round < 0:
            return False

        return completed_round % self._save_every_round == 0

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    def path_for_round(self, completed_round: int) -> Path:
        """
        Return the canonical file path for a round.

        Args:
            completed_round:
                One-based completed round number.

        Returns:
            Absolute path of that round's checkpoint file.

        Raises:
            CheckpointError:
                If the round number is not a non-negative integer.
        """

        if isinstance(completed_round, bool) or not isinstance(
            completed_round, int
        ):
            raise CheckpointError(
                "completed_round must be an integer, got "
                f"{type(completed_round).__name__}."
            )

        if completed_round < 0:
            raise CheckpointError(
                "completed_round must be >= 0, got "
                f"{completed_round}."
            )

        return self._directory / (
            f"{self.FILENAME_PREFIX}"
            f"{completed_round}"
            f"{self.FILENAME_SUFFIX}"
        )

    def available_rounds(self) -> tuple[int, ...]:
        """
        Return every round that currently has a checkpoint.

        Returns:
            Sorted round numbers found on disk. Empty when nothing
            has been saved.
        """

        rounds: list[int] = []

        for path in self._directory.glob(
            f"{self.FILENAME_PREFIX}*{self.FILENAME_SUFFIX}"
        ):
            stem = path.stem

            if not stem.startswith(self.FILENAME_PREFIX):
                continue

            raw = stem[len(self.FILENAME_PREFIX):]

            if not raw.isdigit():
                # Ignore unrelated files that share the pattern.
                continue

            rounds.append(int(raw))

        return tuple(sorted(rounds))

    def exists(self, completed_round: int) -> bool:
        """
        Return whether a checkpoint exists for a round.

        Args:
            completed_round:
                One-based completed round number.

        Returns:
            True if the checkpoint file is present.
        """

        return self.path_for_round(completed_round).is_file()

    def latest_round(self) -> int | None:
        """
        Return the most recent persisted round.

        Returns:
            The highest saved round number, or None when no
            checkpoint has been written.
        """

        rounds = self.available_rounds()

        if not rounds:
            return None

        return rounds[-1]

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    def save(self, checkpoint: FederationCheckpoint) -> Path:
        """
        Persist a checkpoint atomically.

        The checkpoint is serialized into a temporary file in the
        destination directory and then moved into place with
        ``os.replace``. An interrupted save therefore leaves any
        existing checkpoint untouched instead of truncating it.

        Args:
            checkpoint:
                Checkpoint to persist.

        Returns:
            Path of the written checkpoint file.

        Raises:
            CheckpointError:
                If the checkpoint is invalid or cannot be written.
        """

        if not isinstance(checkpoint, FederationCheckpoint):
            raise CheckpointError(
                "save() requires a FederationCheckpoint, got "
                f"{type(checkpoint).__name__}."
            )

        target = self.path_for_round(
            checkpoint.completed_round
        )

        # Shapes of different rank cannot be stored as one ragged
        # array, so shapes are flattened into a single int64 vector
        # and delimited by a per-parameter start offset.
        flat_shapes: list[int] = []
        shape_offsets: list[int] = []

        for spec in checkpoint.contract.specs:
            shape_offsets.append(len(flat_shapes))
            flat_shapes.extend(int(dim) for dim in spec.shape)

        arrays: dict[str, np.ndarray] = {
            "round": np.asarray(
                checkpoint.completed_round,
                dtype=np.int64,
            ),
            "contract_names": np.asarray(
                checkpoint.contract.names,
                dtype=np.str_,
            ),
            "contract_shapes": np.asarray(
                flat_shapes,
                dtype=np.int64,
            ),
            "contract_shape_offsets": np.asarray(
                shape_offsets,
                dtype=np.int64,
            ),
            "contract_dtypes": np.asarray(
                [
                    np.dtype(spec.dtype).str
                    for spec in checkpoint.contract.specs
                ],
                dtype=np.str_,
            ),
        }

        for index, parameter in enumerate(
            checkpoint.parameters
        ):
            arrays[f"param_{index}"] = parameter

        handle = None
        temporary_path: Path | None = None

        try:
            file_descriptor, raw_temporary = tempfile.mkstemp(
                dir=str(self._directory),
                prefix=f".{self.FILENAME_PREFIX}",
                suffix=".tmp",
            )

            temporary_path = Path(raw_temporary)

            with os.fdopen(
                file_descriptor, "wb"
            ) as handle:
                np.savez(handle, **arrays)

                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temporary_path, target)

            temporary_path = None

        except CheckpointError:
            raise

        except Exception as exc:
            raise CheckpointError(
                f"Failed to save checkpoint for round "
                f"{checkpoint.completed_round} to {target}: {exc}"
            ) from exc

        finally:
            if temporary_path is not None:
                # Best-effort cleanup of an incomplete artifact.
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

        return target

    def load(
        self,
        completed_round: int,
    ) -> FederationCheckpoint:
        """
        Load the checkpoint for a specific round.

        Args:
            completed_round:
                One-based completed round number.

        Returns:
            The restored FederationCheckpoint.

        Raises:
            CheckpointError:
                If the checkpoint is missing, unreadable, or does
                not describe a consistent payload.
        """

        path = self.path_for_round(completed_round)

        if not path.is_file():
            raise CheckpointError(
                f"No checkpoint found for round {completed_round} "
                f"at {path}."
            )

        return self._read(path)

    def load_latest(self) -> FederationCheckpoint | None:
        """
        Load the most recent persisted checkpoint.

        Returns:
            The latest FederationCheckpoint, or None when nothing
            has been saved yet.

        Raises:
            CheckpointError:
                If the latest checkpoint is unreadable or
                inconsistent.
        """

        latest = self.latest_round()

        if latest is None:
            return None

        return self.load(latest)

    def delete_round(self, completed_round: int) -> bool:
        """
        Delete a single round's checkpoint.

        Args:
            completed_round:
                One-based completed round number.

        Returns:
            True if a file was removed, False if none existed.
        """

        path = self.path_for_round(completed_round)

        if not path.is_file():
            return False

        try:
            path.unlink()

        except OSError as exc:
            raise CheckpointError(
                f"Failed to delete checkpoint {path}: {exc}"
            ) from exc

        return True

    # --------------------------------------------------------
    # Reading
    # --------------------------------------------------------

    def _read(self, path: Path) -> FederationCheckpoint:
        """Read and validate a checkpoint file."""

        try:
            with np.load(
                path,
                allow_pickle=False,
            ) as archive:
                arrays = {
                    key: archive[key]
                    for key in archive.files
                }

        except CheckpointError:
            raise

        except Exception as exc:
            raise CheckpointError(
                f"Failed to read checkpoint {path}: {exc}"
            ) from exc

        required = {
            "round",
            "contract_names",
            "contract_shapes",
            "contract_shape_offsets",
            "contract_dtypes",
        }

        missing = sorted(required - arrays.keys())

        if missing:
            raise CheckpointError(
                f"Checkpoint {path} is missing required entries: "
                + ", ".join(missing)
            )

        return self._rebuild(path, arrays)

    def _rebuild(
        self,
        path: Path,
        arrays: dict[str, np.ndarray],
    ) -> FederationCheckpoint:
        """Reconstruct a checkpoint from decoded archive arrays."""

        try:
            round_value = int(
                np.asarray(arrays["round"]).reshape(()).item()
            )

        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                f"Checkpoint {path} has an unreadable round entry."
            ) from exc

        names = [str(name) for name in arrays["contract_names"]]
        dtypes = [str(item) for item in arrays["contract_dtypes"]]

        flat_shapes = np.asarray(
            arrays["contract_shapes"],
            dtype=np.int64,
        ).tolist()

        offsets = np.asarray(
            arrays["contract_shape_offsets"],
            dtype=np.int64,
        ).tolist()

        if not (len(names) == len(dtypes) == len(offsets)):
            raise CheckpointError(
                f"Checkpoint {path} has inconsistent contract "
                "metadata lengths."
            )

        specs: list[ParameterSpec] = []

        for index, name in enumerate(names):
            start = int(offsets[index])
            end = (
                int(offsets[index + 1])
                if index + 1 < len(offsets)
                else len(flat_shapes)
            )

            try:
                shape = tuple(
                    int(value) for value in flat_shapes[start:end]
                )
                dtype = np.dtype(dtypes[index])

            except (TypeError, ValueError) as exc:
                raise CheckpointError(
                    f"Checkpoint {path} has unreadable contract "
                    f"metadata for parameter '{name}'."
                ) from exc

            specs.append(
                ParameterSpec(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    is_floating_point=bool(
                        np.issubdtype(dtype, np.floating)
                    ),
                )
            )

        contract = ParameterContract(specs=tuple(specs))

        parameters: ParameterPayload = [
            arrays[f"param_{index}"].copy()
            for index in range(len(specs))
            if f"param_{index}" in arrays
        ]

        if len(parameters) != len(specs):
            raise CheckpointError(
                f"Checkpoint {path} stores {len(parameters)} of "
                f"{len(specs)} expected parameters."
            )

        try:
            return FederationCheckpoint(
                completed_round=round_value,
                contract=contract,
                parameters=parameters,
            )

        except CheckpointError as exc:
            raise CheckpointError(
                f"Checkpoint {path} is inconsistent: {exc}"
            ) from exc


__all__ = [
    "CheckpointError",
    "FederationCheckpoint",
    "FederationCheckpointStore",
]
