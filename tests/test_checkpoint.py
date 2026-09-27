"""
Tests for the framework-independent FedMed checkpoint persistence.

Coverage focuses on the durability boundary while reusing the
established Phase 1/2/3 contracts:

    BaseModel
        ↓
    FederatedClient
        ↓
    FederatedServer
        ↓
    FederationCheckpointStore

Checkpointing is responsible for:

- self-describing checkpoint construction
- payload validation against a recorded contract
- atomic filesystem persistence
- round discovery and latest-round selection
- corrupt-file rejection instead of silent data loss
- save/restore of FederatedServer global state

It does not implement:

- aggregation
- client selection
- local training or evaluation
- round orchestration
- Flower or transport concerns
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from src.aggregation.fedavg import FedAvgAggregator
from src.common.config import CheckpointConfig, TrainingConfig
from src.common.exceptions import FederatedLearningError
from src.fl.checkpoint import (
    CheckpointError,
    FederationCheckpoint,
    FederationCheckpointStore,
)
from src.fl.client import FederatedClient
from src.fl.parameters import (
    ParameterContract,
    ParameterPayload,
)
from src.fl.rounds import RoundCoordinator
from src.fl.server import FederatedServer
from src.fl.strategy import FedAvgStrategy
from src.models.base_model import BaseModel
from src.training.evaluator import Evaluator
from src.training.metrics import Accuracy
from src.training.trainer import Trainer


# ======================================================================
# Test model
# ======================================================================


class CheckpointTestModel(BaseModel):
    """Small deterministic model whose state_dict has mixed ranks.

    ``nn.Linear(2, 2)`` contributes a 2-D weight and a 1-D bias.
    Exercising both is what proves the flattened shape/offset
    encoding round-trips correctly.
    """

    def build(self) -> nn.Module:
        return nn.Linear(2, 2)


class OtherCheckpointTestModel(BaseModel):
    """Incompatible model used to prove contract mismatch detection."""

    def build(self) -> nn.Module:
        return nn.Linear(3, 5)


# ======================================================================
# Test data and client
# ======================================================================


def make_loader(
    size: int = 8,
) -> DataLoader:
    """Create a deterministic DataLoader for a client."""

    torch.manual_seed(42)

    samples = torch.randn(
        size,
        2,
    )

    targets = torch.tensor(
        [0, 1] * (size // 2),
        dtype=torch.long,
    )

    return DataLoader(
        TensorDataset(
            samples,
            targets,
        ),
        batch_size=4,
        shuffle=False,
    )


def make_test_client(
    client_id: str,
) -> FederatedClient:
    """Construct a fully valid FederatedClient."""

    torch.manual_seed(100)

    model = CheckpointTestModel(
        name=f"checkpoint_model_{client_id}",
        device="cpu",
    )

    criterion = nn.CrossEntropyLoss()

    config = TrainingConfig(
        local_epochs=1,
        batch_size=4,
        learning_rate=0.01,
        optimizer="sgd",
        seed=42,
    )

    optimizer = SGD(
        model.parameters(),
        lr=config.learning_rate,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
    )

    evaluator = Evaluator(
        model=model,
        criterion=criterion,
        metrics=[Accuracy()],
    )

    return FederatedClient(
        client_id=client_id,
        model=model,
        trainer=trainer,
        evaluator=evaluator,
        train_loader=make_loader(8),
        eval_loader=make_loader(8),
    )


# ======================================================================
# Federation helpers
# ======================================================================


def make_clients() -> dict[str, FederatedClient]:
    """Create a deterministic two-client federation."""

    return {
        "client_a": make_test_client("client_a"),
        "client_b": make_test_client("client_b"),
    }


def make_strategy() -> FedAvgStrategy:
    """Create the canonical FedAvg strategy."""

    return FedAvgStrategy(
        aggregator=FedAvgAggregator(),
    )


def make_initial_parameters(
    clients: Mapping[str, FederatedClient],
) -> ParameterPayload:
    """Obtain a valid payload from a real client contract."""

    first_client = next(
        iter(clients.values())
    )

    return first_client.get_parameters()


def make_server(
    *,
    clients: Mapping[str, FederatedClient] | None = None,
    strategy: FedAvgStrategy | None = None,
    initial_parameters: ParameterPayload | None = None,
    checkpoint_store: FederationCheckpointStore | None = None,
) -> FederatedServer:
    """Construct a real server, optionally with a store attached."""

    if clients is None:
        clients = make_clients()

    if strategy is None:
        strategy = make_strategy()

    if initial_parameters is None:
        initial_parameters = make_initial_parameters(
            clients
        )

    coordinator = RoundCoordinator(
        strategy=strategy,
        clients=clients,
    )

    return FederatedServer(
        clients=clients,
        strategy=strategy,
        coordinator=coordinator,
        initial_parameters=initial_parameters,
        checkpoint_store=checkpoint_store,
    )


def make_store(
    tmp_path: Path,
    *,
    save_every_round: int = 1,
) -> FederationCheckpointStore:
    """Create a store rooted in a temporary directory."""

    return FederationCheckpointStore(
        tmp_path / "checkpoints",
        save_every_round=save_every_round,
    )


def assert_parameters_equal(
    actual: ParameterPayload,
    expected: ParameterPayload,
) -> None:
    """Compare two payloads tensor-by-tensor."""

    assert len(actual) == len(expected)

    for actual_array, expected_array in zip(
        actual,
        expected,
    ):
        np.testing.assert_array_equal(
            actual_array,
            expected_array,
        )


# ======================================================================
# FederationCheckpoint construction
# ======================================================================


def test_checkpoint_copies_parameters_defensively() -> None:
    """
    A checkpoint must not alias the arrays it was built from.
    """

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    parameters = make_initial_parameters(clients)

    checkpoint = FederationCheckpoint(
        completed_round=1,
        contract=contract,
        parameters=parameters,
    )

    # Mutating the source must not affect the checkpoint.
    parameters[0][0, 0] = 12345.0

    assert checkpoint.parameters[0][
        0, 0
    ] != 12345.0

    # Mutating the checkpoint payload must not affect the source.
    checkpoint.parameters[1][0] = -999.0

    assert parameters[1][0] != -999.0


def test_checkpoint_exposes_parameter_count() -> None:
    """The checkpoint reports its own payload size."""

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    checkpoint = FederationCheckpoint.from_parameters(
        completed_round=2,
        parameters=make_initial_parameters(clients),
        contract=contract,
    )

    assert checkpoint.parameter_count == contract.count
    assert checkpoint.completed_round == 2


def test_checkpoint_accepts_round_zero() -> None:
    """
    Round zero is valid: it represents the initial global model
    captured before any round has run.
    """

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    checkpoint = FederationCheckpoint(
        completed_round=0,
        contract=contract,
        parameters=make_initial_parameters(clients),
    )

    assert checkpoint.completed_round == 0


@pytest.mark.parametrize(
    "completed_round",
    [-1, -5],
)
def test_checkpoint_rejects_negative_round(
    completed_round: int,
) -> None:
    """A negative round number is meaningless and rejected."""

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    with pytest.raises(CheckpointError):
        FederationCheckpoint(
            completed_round=completed_round,
            contract=contract,
            parameters=make_initial_parameters(clients),
        )


@pytest.mark.parametrize(
    "completed_round",
    [True, False, 1.5, "1", None],
)
def test_checkpoint_rejects_non_integer_round(
    completed_round: object,
) -> None:
    """Round numbers must be genuine integers, not bools or strings."""

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    with pytest.raises(CheckpointError):
        FederationCheckpoint(
            completed_round=completed_round,
            contract=contract,
            parameters=make_initial_parameters(clients),
        )


def test_checkpoint_rejects_empty_payload() -> None:
    """An empty payload cannot describe a model."""

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    with pytest.raises(CheckpointError):
        FederationCheckpoint(
            completed_round=1,
            contract=contract,
            parameters=[],
        )


def test_checkpoint_rejects_non_contract() -> None:
    """The contract must be a real ParameterContract."""

    clients = make_clients()

    with pytest.raises(CheckpointError):
        FederationCheckpoint(
            completed_round=1,
            contract="not-a-contract",
            parameters=make_initial_parameters(clients),
        )


def test_checkpoint_rejects_payload_violating_contract() -> None:
    """
    A payload that does not match its own recorded contract must be
    rejected at construction time.
    """

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    parameters = make_initial_parameters(clients)
    parameters[0] = np.zeros(
        (7, 7),
        dtype=np.float32,
    )

    with pytest.raises(CheckpointError):
        FederationCheckpoint(
            completed_round=1,
            contract=contract,
            parameters=parameters,
        )


def test_checkpoint_rejects_non_finite_payload() -> None:
    """
    NaN/Inf values are not a resumable state. The parameter
    contract already rejects them, and the checkpoint must too.
    """

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    parameters = make_initial_parameters(clients)
    parameters[0][0, 0] = np.nan

    with pytest.raises(CheckpointError):
        FederationCheckpoint(
            completed_round=1,
            contract=contract,
            parameters=parameters,
        )


# ======================================================================
# Contract compatibility
# ======================================================================


def test_checkpoint_validates_against_same_contract() -> None:
    """A checkpoint matches the contract it was captured with."""

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    checkpoint = FederationCheckpoint(
        completed_round=1,
        contract=contract,
        parameters=make_initial_parameters(clients),
    )

    checkpoint.validate_against(contract)


def test_checkpoint_rejects_incompatible_contract() -> None:
    """
    A checkpoint captured against a different model must not be
    silently loaded into this federation.
    """

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    checkpoint = FederationCheckpoint(
        completed_round=1,
        contract=contract,
        parameters=make_initial_parameters(clients),
    )

    other_model = OtherCheckpointTestModel(
        name="other_model",
        device="cpu",
    )

    other_contract = ParameterContract.from_model(
        other_model
    )

    with pytest.raises(CheckpointError):
        checkpoint.validate_against(other_contract)


def test_checkpoint_restore_parameters_returns_copies() -> None:
    """Restored parameters must be independent of the checkpoint."""

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    checkpoint = FederationCheckpoint(
        completed_round=1,
        contract=contract,
        parameters=make_initial_parameters(clients),
    )

    restored = checkpoint.restore_parameters()

    restored[0][0, 0] = 4242.0

    assert checkpoint.parameters[0][
        0, 0
    ] != 4242.0


# ======================================================================
# Store construction
# ======================================================================


def test_store_creates_missing_directory(
    tmp_path: Path,
) -> None:
    """
    The store creates its destination directory so callers do not
    have to prepare the filesystem first.
    """

    target = tmp_path / "deeply" / "nested" / "checkpoints"

    store = FederationCheckpointStore(target)

    assert target.is_dir()
    assert store.directory == target


def test_store_reports_configured_interval(
    tmp_path: Path,
) -> None:
    """The configured save interval is readable."""

    store = make_store(
        tmp_path,
        save_every_round=3,
    )

    assert store.save_every_round == 3


@pytest.mark.parametrize(
    "save_every_round",
    [0, -1, True, 1.5, "1", None],
)
def test_store_rejects_invalid_interval(
    tmp_path: Path,
    save_every_round: object,
) -> None:
    """The save interval must be a positive integer."""

    with pytest.raises(CheckpointError):
        FederationCheckpointStore(
            tmp_path / "checkpoints",
            save_every_round=save_every_round,
        )


@pytest.mark.parametrize(
    "directory",
    ["", "   ", None, 5, []],
)
def test_store_rejects_invalid_directory(
    tmp_path: Path,
    directory: object,
) -> None:
    """The directory must be a usable path."""

    with pytest.raises(CheckpointError):
        FederationCheckpointStore(directory)


def test_store_builds_from_checkpoint_config(
    tmp_path: Path,
) -> None:
    """
    The existing CheckpointConfig section becomes consumable.

    This is what gives configs/config.yaml's checkpoint block an
    actual consumer for the first time.
    """

    config = CheckpointConfig(
        enabled=True,
        directory=str(tmp_path / "from_config"),
        save_every_round=2,
    )

    store = FederationCheckpointStore.from_config(
        config
    )

    assert store.directory == Path(
        config.directory
    )
    assert store.save_every_round == 2
    assert Path(config.directory).is_dir()


def test_store_from_config_rejects_foreign_object() -> None:
    """from_config requires a config exposing the right fields."""

    with pytest.raises(CheckpointError):
        FederationCheckpointStore.from_config(
            object()
        )


# ======================================================================
# Save interval
# ======================================================================


@pytest.mark.parametrize(
    "save_every_round,completed_round,expected",
    [
        (1, 0, True),
        (1, 1, True),
        (1, 5, True),
        (2, 0, True),
        (2, 1, False),
        (2, 2, True),
        (2, 3, False),
        (3, 3, True),
        (3, 6, True),
        (3, 7, False),
    ],
)
def test_should_save_respects_interval(
    tmp_path: Path,
    save_every_round: int,
    completed_round: int,
    expected: bool,
) -> None:
    """Only rounds on the interval boundary are persisted."""

    store = make_store(
        tmp_path,
        save_every_round=save_every_round,
    )

    assert store.should_save(completed_round) is expected


@pytest.mark.parametrize(
    "completed_round",
    [-1, True, "2", None, 1.5],
)
def test_should_save_rejects_invalid_round(
    tmp_path: Path,
    completed_round: object,
) -> None:
    """A non-round number never qualifies for saving."""

    store = make_store(tmp_path)

    assert store.should_save(completed_round) is False


# ======================================================================
# Persistence round-trip
# ======================================================================


def test_store_round_trips_checkpoint(
    tmp_path: Path,
) -> None:
    """
    The core guarantee: a saved checkpoint reloads to identical
    parameters, contract, and round number.
    """

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    original = FederationCheckpoint(
        completed_round=3,
        contract=contract,
        parameters=make_initial_parameters(clients),
    )

    store = make_store(tmp_path)

    path = store.save(original)

    assert path.is_file()

    restored = store.load(3)

    assert restored.completed_round == 3
    assert restored.contract.count == contract.count
    assert restored.contract.names == contract.names
    assert restored.contract.shapes == contract.shapes

    assert [
        spec.dtype for spec in restored.contract.specs
    ] == [spec.dtype for spec in contract.specs]

    assert_parameters_equal(
        restored.parameters,
        original.parameters,
    )


def test_store_round_trips_mixed_rank_shapes(
    tmp_path: Path,
) -> None:
    """
    nn.Linear contributes a 2-D weight and a 1-D bias.

    The flattened shape/offset encoding must preserve both ranks
    exactly, otherwise a resumed model would not load.
    """

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    original = FederationCheckpoint(
        completed_round=1,
        contract=contract,
        parameters=make_initial_parameters(clients),
    )

    store = make_store(tmp_path)
    store.save(original)

    restored = store.load(1)

    assert len(restored.contract.shapes) == len(
        original.contract.shapes
    )

    for restored_spec, original_spec in zip(
        restored.contract.specs,
        original.contract.specs,
    ):
        assert restored_spec.shape == original_spec.shape
        assert restored_spec.dtype == original_spec.dtype
        assert (
            restored_spec.is_floating_point
            == original_spec.is_floating_point
        )


def test_store_save_rejects_foreign_object(
    tmp_path: Path,
) -> None:
    """Only a real checkpoint may be persisted."""

    store = make_store(tmp_path)

    with pytest.raises(CheckpointError):
        store.save({"round": 1})


def test_store_load_missing_round_raises(
    tmp_path: Path,
) -> None:
    """
    A missing checkpoint is an error when explicitly requested.

    Silent fallback here is exactly the data-loss mode this
    feature must not have.
    """

    store = make_store(tmp_path)

    with pytest.raises(CheckpointError):
        store.load(4)


def test_store_load_latest_returns_none_when_empty(
    tmp_path: Path,
) -> None:
    """
    "Nothing saved yet" is a normal state for load_latest, unlike
    an explicit load of a specific round.
    """

    store = make_store(tmp_path)

    assert store.load_latest() is None
    assert store.latest_round() is None
    assert store.available_rounds() == ()


# ======================================================================
# Round discovery
# ======================================================================


def test_store_discovers_saved_rounds_in_order(
    tmp_path: Path,
) -> None:
    """Saved rounds are reported sorted, latest last."""

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    store = make_store(tmp_path)

    for round_number in (2, 0, 5, 1):
        store.save(
            FederationCheckpoint(
                completed_round=round_number,
                contract=contract,
                parameters=make_initial_parameters(clients),
            )
        )

    assert store.available_rounds() == (
        0,
        1,
        2,
        5,
    )
    assert store.latest_round() == 5
    assert store.load_latest().completed_round == 5


def test_store_ignores_unrelated_files(
    tmp_path: Path,
) -> None:
    """
    Foreign files sharing the naming pattern must not be mistaken
    for checkpoints.
    """

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    store = make_store(tmp_path)

    store.save(
        FederationCheckpoint(
            completed_round=1,
            contract=contract,
            parameters=make_initial_parameters(clients),
        )
    )

    (store.directory / "round_abc.npz").write_bytes(
        b"junk"
    )
    (store.directory / "round_1.txt").write_bytes(
        b"junk"
    )
    (store.directory / "notes.txt").write_bytes(
        b"junk"
    )

    assert store.available_rounds() == (1,)
    assert store.latest_round() == 1


def test_store_exists_reflects_presence(
    tmp_path: Path,
) -> None:
    """exists() answers whether a specific round is present."""

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    store = make_store(tmp_path)

    assert store.exists(1) is False

    store.save(
        FederationCheckpoint(
            completed_round=1,
            contract=contract,
            parameters=make_initial_parameters(clients),
        )
    )

    assert store.exists(1) is True


def test_store_path_for_round_is_canonical(
    tmp_path: Path,
) -> None:
    """Round-to-path mapping is stable and predictable."""

    store = make_store(tmp_path)

    assert store.path_for_round(7) == (
        store.directory / "round_7.npz"
    )


@pytest.mark.parametrize(
    "completed_round",
    [-1, True, "1", None],
)
def test_store_path_rejects_invalid_round(
    tmp_path: Path,
    completed_round: object,
) -> None:
    """Path derivation rejects invalid round numbers."""

    store = make_store(tmp_path)

    with pytest.raises(CheckpointError):
        store.path_for_round(completed_round)


def test_store_delete_round(
    tmp_path: Path,
) -> None:
    """A single checkpoint can be removed, idempotently."""

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    store = make_store(tmp_path)

    store.save(
        FederationCheckpoint(
            completed_round=1,
            contract=contract,
            parameters=make_initial_parameters(clients),
        )
    )

    assert store.delete_round(1) is True
    assert store.delete_round(1) is False
    assert store.exists(1) is False


# ======================================================================
# Corruption handling
# ======================================================================


def test_store_rejects_corrupt_file(
    tmp_path: Path,
) -> None:
    """
    A truncated file must raise rather than resume from garbage.
    """

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    store = make_store(tmp_path)

    path = store.save(
        FederationCheckpoint(
            completed_round=1,
            contract=contract,
            parameters=make_initial_parameters(clients),
        )
    )

    path.write_bytes(b"this is not an npz archive")

    with pytest.raises(CheckpointError):
        store.load(1)

    with pytest.raises(CheckpointError):
        store.load_latest()


def test_store_rejects_incomplete_archive(
    tmp_path: Path,
) -> None:
    """
    An archive missing required contract metadata is rejected.

    Without the contract a checkpoint cannot validate itself, so
    accepting it would defeat the point of persisting it.
    """

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    store = make_store(tmp_path)

    path = store.path_for_round(1)

    np.savez(
        path,
        round=np.asarray(1, dtype=np.int64),
        contract_names=np.asarray(
            contract.names,
            dtype=np.str_,
        ),
    )

    with pytest.raises(CheckpointError):
        store.load(1)


def test_store_rejects_missing_parameter_entries(
    tmp_path: Path,
) -> None:
    """
    A checkpoint declaring more parameters than it stores is
    inconsistent and must not load.
    """

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    store = make_store(tmp_path)

    path = store.path_for_round(1)

    np.savez(
        path,
        round=np.asarray(1, dtype=np.int64),
        contract_names=np.asarray(
            contract.names,
            dtype=np.str_,
        ),
        contract_shapes=np.asarray(
            [dim for spec in contract.specs for dim in spec.shape],
            dtype=np.int64,
        ),
        contract_shape_offsets=np.asarray(
            np.cumsum(
                [0]
                + [len(spec.shape) for spec in contract.specs]
            )[:-1],
            dtype=np.int64,
        ),
        contract_dtypes=np.asarray(
            [np.dtype(spec.dtype).str for spec in contract.specs],
            dtype=np.str_,
        ),
        param_0=np.zeros(
            (2, 2),
            dtype=np.float32,
        ),
    )

    with pytest.raises(CheckpointError):
        store.load(1)


# ======================================================================
# Atomicity
# ======================================================================


def test_store_save_is_atomic_and_leaves_no_temp_files(
    tmp_path: Path,
) -> None:
    """
    A completed save leaves no temporary artifacts behind.
    """

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    store = make_store(tmp_path)

    store.save(
        FederationCheckpoint(
            completed_round=1,
            contract=contract,
            parameters=make_initial_parameters(clients),
        )
    )

    files = sorted(
        path.name
        for path in store.directory.iterdir()
    )

    assert files == ["round_1.npz"]


def test_store_failed_save_preserves_existing_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A crash during serialization must not damage a good
    checkpoint, because the write is staged then atomically
    replaced.
    """

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    store = make_store(tmp_path)

    good = FederationCheckpoint(
        completed_round=1,
        contract=contract,
        parameters=make_initial_parameters(clients),
    )

    store.save(good)

    replacement = FederationCheckpoint(
        completed_round=2,
        contract=contract,
        parameters=make_initial_parameters(clients),
    )

    def exploding_savez(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(
        np,
        "savez",
        exploding_savez,
    )

    with pytest.raises(CheckpointError):
        store.save(replacement)

    monkeypatch.undo()

    # The pre-existing checkpoint is still intact and loadable.
    restored = store.load(1)

    assert restored.completed_round == 1
    assert_parameters_equal(
        restored.parameters,
        good.parameters,
    )

    # Round 2 was never created, and no temp file leaked.
    assert store.exists(2) is False

    files = sorted(
        path.name
        for path in store.directory.iterdir()
    )

    assert files == ["round_1.npz"]


def test_store_save_overwrites_existing_round(
    tmp_path: Path,
) -> None:
    """Re-saving a round replaces it atomically."""

    clients = make_clients()
    contract = next(
        iter(clients.values())
    ).parameter_contract

    store = make_store(tmp_path)

    store.save(
        FederationCheckpoint(
            completed_round=1,
            contract=contract,
            parameters=make_initial_parameters(clients),
        )
    )

    updated_parameters = make_initial_parameters(clients)
    updated_parameters[0][0, 0] = 77.0

    store.save(
        FederationCheckpoint(
            completed_round=1,
            contract=contract,
            parameters=updated_parameters,
        )
    )

    restored = store.load(1)

    assert restored.parameters[0][
        0, 0
    ] == pytest.approx(77.0)


# ======================================================================
# Server integration: opt-in behavior
# ======================================================================


def test_server_without_store_has_no_store(
    tmp_path: Path,
) -> None:
    """
    Omitting the store preserves prior behavior exactly: the
    server reports no store and touches no filesystem.
    """

    server = make_server()

    assert server.checkpoint_store is None


def test_server_runs_normally_without_store(
    tmp_path: Path,
) -> None:
    """
    Round execution without a store is unchanged and writes
    nothing to the checkpoint directory.
    """

    target = tmp_path / "unused"

    server = make_server()

    server.run_round()

    assert server.completed_round == 1
    assert target.exists() is False


def test_server_exposes_parameter_contract() -> None:
    """
    The derived contract matches the participating clients'.
    """

    clients = make_clients()

    server = make_server(clients=clients)

    first_contract = next(
        iter(clients.values())
    ).parameter_contract

    assert (
        server.parameter_contract.count
        == first_contract.count
    )
    assert (
        server.parameter_contract.names
        == first_contract.names
    )


def test_server_rejects_foreign_checkpoint_store(
    tmp_path: Path,
) -> None:
    """The store must be a real FederationCheckpointStore."""

    with pytest.raises(FederatedLearningError):
        make_server(
            checkpoint_store="not-a-store"
        )


def test_server_save_requires_store() -> None:
    """
    Calling save without a store fails clearly instead of
    silently doing nothing.
    """

    server = make_server()

    with pytest.raises(FederatedLearningError):
        server.save_checkpoint()


def test_server_restore_requires_store() -> None:
    """Calling restore without a store fails clearly."""

    server = make_server()

    with pytest.raises(FederatedLearningError):
        server.restore_checkpoint()


# ======================================================================
# Server integration: automatic saving
# ======================================================================


def test_server_saves_after_each_round(
    tmp_path: Path,
) -> None:
    """
    With an interval of 1, every committed round is persisted.
    """

    store = make_store(
        tmp_path,
        save_every_round=1,
    )

    server = make_server(checkpoint_store=store)

    server.run_round()
    server.run_round()

    assert server.completed_round == 2
    assert store.available_rounds() == (1, 2)
    assert store.latest_round() == 2


def test_server_respects_save_interval(
    tmp_path: Path,
) -> None:
    """
    With an interval of 2, only even rounds are persisted.
    """

    store = make_store(
        tmp_path,
        save_every_round=2,
    )

    server = make_server(checkpoint_store=store)

    server.run_round()
    server.run_round()
    server.run_round()

    assert server.completed_round == 3
    assert store.available_rounds() == (2,)


def test_server_saved_checkpoint_matches_server_state(
    tmp_path: Path,
) -> None:
    """
    The persisted checkpoint captures the post-aggregation global
    model, not the pre-round parameters.
    """

    store = make_store(tmp_path)

    server = make_server(checkpoint_store=store)

    execution = server.run_round()

    restored = store.load(1)

    assert restored.completed_round == 1

    assert_parameters_equal(
        restored.parameters,
        execution.aggregated_parameters,
    )

    assert_parameters_equal(
        restored.parameters,
        server.global_parameters,
    )


def test_server_multi_run_saves_each_round(
    tmp_path: Path,
) -> None:
    """run(num_rounds=...) persists each committed round."""

    store = make_store(tmp_path)

    server = make_server(checkpoint_store=store)

    server.run(num_rounds=3)

    assert store.available_rounds() == (1, 2, 3)


# ======================================================================
# Server integration: manual save and restore
# ======================================================================


def test_server_manual_save_and_restore_round_trip(
    tmp_path: Path,
) -> None:
    """
    The headline capability: a federation runs, persists, and a
    fresh server resumes from the exact global model and round.
    """

    store = make_store(tmp_path)

    source = make_server(checkpoint_store=store)

    source.run(num_rounds=2)

    expected_parameters = source.global_parameters

    checkpoint = source.save_checkpoint()

    assert checkpoint.completed_round == 2

    # A brand-new server, as after a process restart.
    restored_server = make_server(
        checkpoint_store=store
    )

    assert restored_server.completed_round == 0

    loaded = restored_server.restore_checkpoint()

    assert loaded.completed_round == 2
    assert restored_server.completed_round == 2

    assert_parameters_equal(
        restored_server.global_parameters,
        expected_parameters,
    )


def test_restored_server_continues_numbering(
    tmp_path: Path,
) -> None:
    """
    A restored server continues strictly contiguous rounds rather
    than restarting at one.
    """

    store = make_store(tmp_path)

    source = make_server(checkpoint_store=store)
    source.run(num_rounds=2)

    resumed = make_server(checkpoint_store=store)
    resumed.restore_checkpoint()

    execution = resumed.run_round()

    assert execution.result.round_number == 3
    assert resumed.completed_round == 3

    with pytest.raises(FederatedLearningError):
        resumed.run_round(round_number=1)


def test_restore_rejects_mismatched_federation(
    tmp_path: Path,
) -> None:
    """
    A checkpoint from a different model must not be loaded.

    The server must be left completely untouched.
    """

    store = make_store(tmp_path)

    source = make_server(checkpoint_store=store)
    source.run_round()

    # A federation whose model has a different parameter layout.
    def make_incompatible_client(
        client_id: str,
    ) -> FederatedClient:
        torch.manual_seed(100)

        model = OtherCheckpointTestModel(
            name=f"other_{client_id}",
            device="cpu",
        )

        config = TrainingConfig(
            local_epochs=1,
            batch_size=4,
            learning_rate=0.01,
            optimizer="sgd",
            seed=42,
        )

        criterion = nn.CrossEntropyLoss()

        return FederatedClient(
            client_id=client_id,
            model=model,
            trainer=Trainer(
                model=model,
                criterion=criterion,
                optimizer=SGD(
                    model.parameters(),
                    lr=config.learning_rate,
                ),
                config=config,
            ),
            evaluator=Evaluator(
                model=model,
                criterion=criterion,
                metrics=[Accuracy()],
            ),
            train_loader=make_loader(8),
            eval_loader=make_loader(8),
        )

    incompatible_clients = {
        "client_x": make_incompatible_client("client_x"),
    }

    strategy = make_strategy()

    incompatible_server = FederatedServer(
        clients=incompatible_clients,
        strategy=strategy,
        coordinator=RoundCoordinator(
            strategy=strategy,
            clients=incompatible_clients,
        ),
        initial_parameters=next(
            iter(incompatible_clients.values())
        ).get_parameters(),
        checkpoint_store=store,
    )

    before = incompatible_server.global_parameters

    with pytest.raises(FederatedLearningError):
        incompatible_server.restore_checkpoint()

    # Rejected restore must not mutate server state.
    assert incompatible_server.completed_round == 0
    assert_parameters_equal(
        incompatible_server.global_parameters,
        before,
    )


def test_restore_specific_round(
    tmp_path: Path,
) -> None:
    """A specific historical round can be restored."""

    store = make_store(tmp_path)

    source = make_server(checkpoint_store=store)
    source.run(num_rounds=3)

    resumed = make_server(checkpoint_store=store)

    loaded = resumed.restore_checkpoint(2)

    assert loaded.completed_round == 2
    assert resumed.completed_round == 2


def test_restore_without_any_checkpoint_raises(
    tmp_path: Path,
) -> None:
    """
    Resuming from an empty store is an error, not a silent reset.

    Silently starting from scratch would discard progress without
    telling anyone.
    """

    store = make_store(tmp_path)

    server = make_server(checkpoint_store=store)

    with pytest.raises(FederatedLearningError):
        server.restore_checkpoint()


def test_restore_missing_round_raises(
    tmp_path: Path,
) -> None:
    """Restoring a round that was never saved fails clearly."""

    store = make_store(tmp_path)

    server = make_server(checkpoint_store=store)
    server.run_round()

    with pytest.raises(FederatedLearningError):
        server.restore_checkpoint(9)


def test_restore_resets_round_history(
    tmp_path: Path,
) -> None:
    """
    Durable state is the global model plus round progression.

    RoundExecution carries live client results and per-round
    metrics that are not part of the durable state, so history is
    intentionally empty after a restore.
    """

    store = make_store(tmp_path)

    source = make_server(checkpoint_store=store)
    source.run(num_rounds=2)

    assert len(source.round_history) == 2

    resumed = make_server(checkpoint_store=store)
    resumed.restore_checkpoint()

    assert resumed.round_history == ()
    assert resumed.completed_round == 2


def test_restore_leaves_source_untouched(
    tmp_path: Path,
) -> None:
    """
    Restoring into one server must not disturb the server that
    produced the checkpoint.
    """

    store = make_store(tmp_path)

    source = make_server(checkpoint_store=store)
    source.run_round()

    expected = source.global_parameters
    history_length = len(source.round_history)

    resumed = make_server(checkpoint_store=store)
    resumed.restore_checkpoint()
    resumed.run_round()

    assert source.completed_round == 1
    assert len(source.round_history) == history_length
    assert_parameters_equal(
        source.global_parameters,
        expected,
    )


def test_failed_round_does_not_advance_checkpoint(
    tmp_path: Path,
) -> None:
    """
    A round that fails must not produce a checkpoint.

    Persistence happens only after a committed round, so a failure
    can never be recorded as progress.
    """

    from src.common.exceptions import (
        FederatedLearningError as FLFederatedLearningError,
    )

    store = make_store(tmp_path)

    server = make_server(checkpoint_store=store)

    class ExplodingCoordinator(RoundCoordinator):
        def execute_round(self, *args, **kwargs):
            raise FLFederatedLearningError(
                "synthetic round failure"
            )

    clients = dict(server.clients)
    strategy = server.strategy

    failing_server = FederatedServer(
        clients=clients,
        strategy=strategy,
        coordinator=ExplodingCoordinator(
            strategy=strategy,
            clients=clients,
        ),
        initial_parameters=make_initial_parameters(
            clients
        ),
        checkpoint_store=store,
    )

    with pytest.raises(FederatedLearningError):
        failing_server.run_round()

    assert failing_server.completed_round == 0
    assert store.available_rounds() == ()


# ======================================================================
# Framework independence
# ======================================================================


def test_checkpoint_module_is_framework_independent() -> None:
    """
    src.fl.checkpoint must not import Flower/runtime APIs.

    Runtime adapters belong in app/.
    """

    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "fl"
        / "checkpoint.py"
    ).read_text(encoding="utf-8")

    forbidden_imports = (
        "import flwr",
        "from flwr",
        "ServerApp",
        "ClientApp",
    )

    for forbidden in forbidden_imports:
        assert forbidden not in source


def test_checkpoint_error_is_federated_error() -> None:
    """
    CheckpointError must stay inside the existing FedMed federated
    exception hierarchy.
    """

    assert issubclass(
        CheckpointError,
        FederatedLearningError,
    )
