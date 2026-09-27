"""
Federated server orchestration for FedMed.

Phase 3.4-A
-----------

This module establishes the framework-independent server-side
federation lifecycle.

Architecture
------------

                    FederatedServer
                           |
              +------------+------------+
              |            |            |
              v            v            v
       Global state    Strategy    RoundCoordinator
                                          |
                                          v
                                  FederatedClient
                                          |
                                          v
                                    local training
                                          |
                                          v
                                      Strategy
                                          |
                                          v
                                      Aggregator
                                          |
                                          v
                                  Global parameters


Responsibilities
----------------

FederatedServer owns:

- current global ParameterPayload
- registered federated clients
- federation strategy dependency
- RoundCoordinator dependency
- completed-round progression
- immutable round execution history
- multi-round orchestration

FederatedServer intentionally does NOT:

- implement local training
- implement local evaluation
- select clients itself
- implement aggregation mathematics
- implement FedAvg
- implement metric aggregation
- implement networking or transport
- import Flower/runtime APIs
- implement checkpoint persistence
- implement retry/timeout policy

Those responsibilities remain behind the existing FedMed
contracts and future runtime/operational layers.

Checkpoint persistence
----------------------
The server does not implement checkpoint serialization itself. When
an optional ``FederationCheckpointStore`` is attached, the server
delegates persistence of its own long-lived state to that store
after each committed round, and can restore itself from a
persisted checkpoint.

Omitting the store leaves behavior unchanged: no filesystem access
and no state is written. The actual serialization format lives in
``src/fl/checkpoint.py``.

Design invariants
-----------------

1. Global parameters are defensively copied.
2. Global parameters are validated against every registered
   client's ParameterContract during construction.
3. Client identity and coordinator/client-registry consistency
   are validated during construction.
4. Round numbers are one-based and strictly contiguous.
5. A completed round updates global parameters exactly once.
6. A failed round never replaces current global parameters.
7. RoundExecution objects are retained in immutable history.
8. Returned parameters/history are defensive or read-only views.
9. The server delegates one-round execution to RoundCoordinator.
10. Strategy and Aggregator responsibilities remain outside the
    server.

This module is deliberately framework-independent. Runtime-specific
adapters belong in the application/runtime boundary, such as
``app/server.py``.
"""

from __future__ import annotations

from collections.abc import Mapping

from types import MappingProxyType

from src.common.exceptions import FederatedLearningError
from src.fl.checkpoint import (
    FederationCheckpoint,
    FederationCheckpointStore,
)
from src.fl.client import FederatedClient
from src.fl.parameters import (
    ParameterContract,
    ParameterPayload,
    copy_parameters,
    validate_contract_compatibility,
    validate_parameters,
)
from src.fl.rounds import RoundCoordinator, RoundExecution
from src.fl.strategy import FederatedStrategy


class FederatedServer:
    """
    Server-side owner of a FedMed federation session.

    The server coordinates multiple rounds while delegating the
    execution of each individual round to ``RoundCoordinator``.

    Parameters
    ----------
    clients:
        Stable mapping of client IDs to ``FederatedClient`` objects.

    strategy:
        Federation policy used by the injected coordinator.

    coordinator:
        Reusable ``RoundCoordinator`` responsible for executing
        exactly one federated round.

    initial_parameters:
        Initial global model state represented by FedMed's
        framework-independent ``ParameterPayload``.

    checkpoint_store:
        Optional ``FederationCheckpointStore`` used to persist and
        restore the server's long-lived state.

        When provided, the server saves a checkpoint after each
        committed round that falls on the store's configured
        interval. When omitted, the server performs no persistence
        and its behavior is identical to earlier versions.

    Notes
    -----
    ``clients`` and ``coordinator.clients`` must describe the same
    federation. The server rejects divergent registries rather than
    allowing two sources of participant truth.
    """

    def __init__(
        self,
        clients: Mapping[str, FederatedClient],
        strategy: FederatedStrategy,
        coordinator: RoundCoordinator,
        initial_parameters: ParameterPayload,
        *,
        checkpoint_store: FederationCheckpointStore | None = None,
    ) -> None:
        # --------------------------------------------------------
        # Validate dependencies
        # --------------------------------------------------------

        self._validate_clients(
            clients,
        )

        if not isinstance(
            strategy,
            FederatedStrategy,
        ):
            raise FederatedLearningError(
                "FederatedServer.strategy must be a "
                "FederatedStrategy."
            )

        if not isinstance(
            coordinator,
            RoundCoordinator,
        ):
            raise FederatedLearningError(
                "FederatedServer.coordinator must be a "
                "RoundCoordinator."
            )

        if checkpoint_store is not None and not isinstance(
            checkpoint_store,
            FederationCheckpointStore,
        ):
            raise FederatedLearningError(
                "FederatedServer.checkpoint_store must be a "
                "FederationCheckpointStore or None, got "
                f"{type(checkpoint_store).__name__}."
            )

        normalized_clients = dict(
            clients,
        )

        # --------------------------------------------------------
        # Cross-component consistency
        # --------------------------------------------------------

        self._validate_coordinator_consistency(
            normalized_clients,
            strategy,
            coordinator,
        )

        # --------------------------------------------------------
        # Initial global state
        # --------------------------------------------------------

        copied_parameters = (
            self._validate_and_copy_parameters(
                initial_parameters,
                normalized_clients,
            )
        )

        # --------------------------------------------------------
        # Long-lived federation state
        # --------------------------------------------------------

        self._clients = MappingProxyType(
            normalized_clients,
        )

        self._strategy = strategy

        self._coordinator = coordinator

        # The server owns this defensive copy.
        self._global_parameters = copied_parameters

        # No federated round has completed yet.
        #
        # FedMed rounds are one-based, therefore zero is the
        # natural "nothing completed" state.
        self._completed_round = 0

        # Immutable history of successfully completed rounds.
        self._round_history: tuple[
            RoundExecution,
            ...
        ] = ()

        # Optional durability boundary.
        #
        # ``None`` means this server never touches the filesystem.
        self._checkpoint_store = checkpoint_store

        # Canonical parameter layout shared by the federation.
        #
        # Derived once so a checkpoint can describe itself without
        # re-deriving the layout on every save.
        self._parameter_contract = (
            self._derive_parameter_contract(
                normalized_clients,
            )
        )

    # ============================================================
    # Public properties
    # ============================================================

    @property
    def clients(
        self,
    ) -> Mapping[str, FederatedClient]:
        """
        Return the registered clients as a read-only mapping.
        """

        return self._clients

    @property
    def strategy(
        self,
    ) -> FederatedStrategy:
        """
        Return the configured federation strategy.
        """

        return self._strategy

    @property
    def coordinator(
        self,
    ) -> RoundCoordinator:
        """
        Return the configured round coordinator.
        """

        return self._coordinator

    @property
    def global_parameters(
        self,
    ) -> ParameterPayload:
        """
        Return a defensive copy of the current global parameters.

        The server never exposes its internal mutable NumPy arrays
        directly.
        """

        return copy_parameters(
            self._global_parameters,
        )

    @property
    def completed_round(
        self,
    ) -> int:
        """
        Return the number of the most recently completed round.

        Returns zero before any round has completed.
        """

        return self._completed_round

    @property
    def round_history(
        self,
    ) -> tuple[RoundExecution, ...]:
        """
        Return the immutable history of completed rounds.
        """

        return self._round_history

    @property
    def checkpoint_store(
        self,
    ) -> FederationCheckpointStore | None:
        """
        Return the attached checkpoint store, if any.

        ``None`` means the server performs no persistence.
        """

        return self._checkpoint_store

    @property
    def parameter_contract(self) -> ParameterContract:
        """
        Return the federation's canonical parameter layout.

        Every registered client is verified to share this layout,
        which is what allows a checkpoint to describe itself.
        """

        return self._parameter_contract

    # ============================================================
    # Checkpointing
    # ============================================================

    def save_checkpoint(self) -> FederationCheckpoint | None:
        """
        Persist the current global state through the attached store.

        Returns:
            The written FederationCheckpoint, or None when no
            store is attached.

        Raises:
            FederatedLearningError:
                If no store is attached, or the checkpoint could
                not be written.
        """

        store = self._require_checkpoint_store(
            "save_checkpoint"
        )

        return self._write_checkpoint(store)

    def restore_checkpoint(
        self,
        completed_round: int | None = None,
    ) -> FederationCheckpoint:
        """
        Restore global state from a persisted checkpoint.

        The checkpoint's recorded contract is validated against the
        federation's canonical contract, and its payload is
        validated against every registered client, before any
        server state is changed. A rejected checkpoint therefore
        leaves the server untouched.

        Round history is intentionally not restored. ``RoundExecution``
        carries live client result objects and per-round metrics
        that are not part of the durable state; the restored server
        reports an empty history while resuming at
        ``completed_round``.

        Parameters:
            completed_round:
                Round to restore. When omitted, the store's most
                recent checkpoint is used.

        Returns:
            The restored FederationCheckpoint.

        Raises:
            FederatedLearningError:
                If no store is attached, no checkpoint exists, or
                the checkpoint does not belong to this federation.
        """

        store = self._require_checkpoint_store(
            "restore_checkpoint"
        )

        if completed_round is None:
            checkpoint = store.load_latest()

            if checkpoint is None:
                raise FederatedLearningError(
                    "No checkpoint is available to restore from "
                    f"{store.directory}."
                )

        else:
            checkpoint = store.load(completed_round)

        # --------------------------------------------------------
        # Validate everything BEFORE mutating server state.
        # --------------------------------------------------------

        try:
            checkpoint.validate_against(
                self._parameter_contract,
            )

        except FederatedLearningError as exc:
            raise FederatedLearningError(
                "Refusing to restore a checkpoint that does not "
                f"match this federation: {exc}"
            ) from exc

        restored_parameters = (
            self._validate_and_copy_parameters(
                checkpoint.parameters,
                self._clients,
            )
        )

        # --------------------------------------------------------
        # Atomic commit of restored state
        # --------------------------------------------------------

        self._global_parameters = restored_parameters

        self._completed_round = (
            checkpoint.completed_round
        )

        # Durable state is the global model and round progression.
        self._round_history = ()

        return checkpoint

    def _require_checkpoint_store(
        self,
        operation: str,
    ) -> FederationCheckpointStore:
        """
        Return the attached store or fail clearly.
        """

        if self._checkpoint_store is None:
            raise FederatedLearningError(
                f"{operation}() requires a checkpoint store, but "
                "FederatedServer was constructed without one."
            )

        return self._checkpoint_store

    def _write_checkpoint(
        self,
        store: FederationCheckpointStore,
    ) -> FederationCheckpoint:
        """
        Build and persist a checkpoint of current global state.
        """

        checkpoint = FederationCheckpoint(
            completed_round=self._completed_round,
            contract=self._parameter_contract,
            parameters=self._global_parameters,
        )

        store.save(checkpoint)

        return checkpoint

    # ============================================================
    # Single-round execution
    # ============================================================

    def run_round(
        self,
        round_number: int | None = None,
        *,
        evaluate: bool = False,
    ) -> RoundExecution:
        """
        Execute exactly one next federated round.

        Parameters
        ----------
        round_number:
            Optional explicit one-based round number.

            If omitted, the server executes the next expected
            round.

            If supplied, it must equal:

                completed_round + 1

            Strict contiguity prevents accidental skipped rounds.

        evaluate:
            Whether the RoundCoordinator should execute its
            optional evaluation phase.

        Returns
        -------
        RoundExecution
            Immutable operational result of the completed round.

        Raises
        ------
        FederatedLearningError
            If the requested round is invalid, the coordinator
            fails, or the returned execution violates the server
            contract.

        Failure semantics
        -----------------
        If the coordinator raises an exception, the server's:

        - global parameters
        - completed round
        - round history

        remain unchanged.
        """

        expected_round = (
            self._completed_round + 1
        )

        if round_number is None:
            round_number = expected_round

        self._validate_next_round_number(
            round_number,
            expected_round,
        )

        # Never give the coordinator direct access to the server's
        # internal parameter arrays.
        round_parameters = copy_parameters(
            self._global_parameters,
        )

        try:
            execution = (
                self._coordinator.execute_round(
                    round_number=round_number,
                    parameters=round_parameters,
                    evaluate=evaluate,
                )
            )

        except FederatedLearningError:
            # Deliberately preserve the existing domain exception.
            #
            # Retry/abort policy belongs above the core server
            # lifecycle.
            raise

        except Exception as exc:
            raise FederatedLearningError(
                f"Federated round {round_number} failed "
                f"unexpectedly: {exc}"
            ) from exc

        # --------------------------------------------------------
        # Validate the complete result BEFORE changing server state.
        # --------------------------------------------------------

        self._validate_round_execution(
            execution,
            expected_round=expected_round,
        )

        # --------------------------------------------------------
        # Atomic server-state commit
        # --------------------------------------------------------

        new_global_parameters = (
            copy_parameters(
                execution.aggregated_parameters,
            )
        )

        self._global_parameters = (
            new_global_parameters
        )

        self._completed_round = (
            round_number
        )

        self._round_history = (
            *self._round_history,
            execution,
        )

        # --------------------------------------------------------
        # Optional durability
        # --------------------------------------------------------
        #
        # This runs only after the round is fully committed, so a
        # failed save can never roll back or corrupt a completed
        # round. The round stays committed and the error surfaces
        # to the caller.

        if self._checkpoint_store is not None:
            if self._checkpoint_store.should_save(
                self._completed_round
            ):
                self._write_checkpoint(
                    self._checkpoint_store,
                )

        return execution

    # ============================================================
    # Multi-round execution
    # ============================================================

    def run(
        self,
        num_rounds: int,
        *,
        evaluate: bool = False,
    ) -> tuple[RoundExecution, ...]:
        """
        Execute a consecutive sequence of federated rounds.

        ``num_rounds`` means the number of additional rounds to
        execute, not the final absolute round number.

        Example
        -------

        If:

            completed_round == 2

        then:

            server.run(num_rounds=3)

        executes:

            round 3
            round 4
            round 5

        Returns
        -------
        tuple[RoundExecution, ...]
            Results produced by this invocation.

        Failure semantics
        -----------------
        If a later round fails, previously completed rounds from
        this invocation remain committed.

        The failed round itself does not modify global state.
        """

        self._validate_num_rounds(
            num_rounds,
        )

        executions: list[
            RoundExecution
        ] = []

        for _ in range(num_rounds):
            execution = self.run_round(
                evaluate=evaluate,
            )

            executions.append(
                execution,
            )

        return tuple(
            executions,
        )

    # ============================================================
    # Client validation
    # ============================================================

    @staticmethod
    def _validate_clients(
        clients: Mapping[str, FederatedClient],
    ) -> None:
        """
        Validate the server's client registry.
        """

        if not isinstance(
            clients,
            Mapping,
        ):
            raise FederatedLearningError(
                "FederatedServer.clients must be a mapping."
            )

        if not clients:
            raise FederatedLearningError(
                "FederatedServer requires at least one client."
            )

        for client_id, client in clients.items():

            if not isinstance(
                client_id,
                str,
            ):
                raise FederatedLearningError(
                    "FederatedServer client IDs must be strings."
                )

            if not client_id.strip():
                raise FederatedLearningError(
                    "FederatedServer client IDs cannot be empty."
                )

            if not isinstance(
                client,
                FederatedClient,
            ):
                raise FederatedLearningError(
                    f"Client '{client_id}' must be a "
                    "FederatedClient, "
                    f"got {type(client).__name__}."
                )

            if client.client_id != client_id:
                raise FederatedLearningError(
                    "FederatedServer client registry key "
                    "does not match the client's own ID: "
                    f"key='{client_id}', "
                    f"client_id='{client.client_id}'."
                )

    # ============================================================
    # Coordinator consistency
    # ============================================================

    @staticmethod
    def _validate_coordinator_consistency(
        clients: Mapping[str, FederatedClient],
        strategy: FederatedStrategy,
        coordinator: RoundCoordinator,
    ) -> None:
        """
        Ensure the server and coordinator represent the same
        federation.

        The coordinator already validates its own client registry.
        The server additionally prevents two conflicting sources
        of federation state.
        """

        coordinator_clients = (
            coordinator.clients
        )

        if tuple(clients.keys()) != tuple(
            coordinator_clients.keys()
        ):
            raise FederatedLearningError(
                "FederatedServer.clients and "
                "RoundCoordinator.clients must contain "
                "the same client IDs in the same deterministic "
                "order."
            )

        for client_id, client in clients.items():

            if (
                coordinator_clients[client_id]
                is not client
            ):
                raise FederatedLearningError(
                    "FederatedServer and RoundCoordinator "
                    "must reference the same "
                    "FederatedClient instance for "
                    f"client '{client_id}'."
                )

        if coordinator.strategy is not strategy:
            raise FederatedLearningError(
                "FederatedServer.strategy must be the same "
                "strategy instance configured in the "
                "RoundCoordinator."
            )

    # ============================================================
    # Checkpoint support
    # ============================================================

    @staticmethod
    def _derive_parameter_contract(
        clients: Mapping[str, FederatedClient],
    ) -> ParameterContract:
        """
        Derive the federation's canonical parameter layout.

        Every registered client must describe an identical layout.
        A federation whose clients disagree cannot produce a
        self-describing checkpoint, so divergence is rejected up
        front rather than at save time.

        Phase 3.1 remains the canonical contract-comparison
        boundary.
        """

        contracts = [
            client.parameter_contract
            for client in clients.values()
        ]

        if not contracts:
            raise FederatedLearningError(
                "Cannot derive a parameter contract without clients."
            )

        canonical = contracts[0]

        for client_id, contract in zip(
            clients, contracts[1:]
        ):
            try:
                validate_contract_compatibility(
                    canonical,
                    contract,
                )

            except FederatedLearningError as exc:
                raise FederatedLearningError(
                    "Client '"
                    f"{client_id}' declares a parameter contract "
                    f"incompatible with the rest of the federation: "
                    f"{exc}"
                ) from exc

        return canonical

    # ============================================================
    # Initial parameter validation
    # ============================================================

    @staticmethod
    def _validate_and_copy_parameters(
        parameters: ParameterPayload,
        clients: Mapping[str, FederatedClient],
    ) -> ParameterPayload:
        """
        Validate initial global parameters against every client's
        ParameterContract and return a defensive copy.

        Phase 3.1 remains the canonical parameter-validation
        boundary.
        """

        if parameters is None:
            raise FederatedLearningError(
                "FederatedServer.initial_parameters "
                "cannot be None."
            )

        try:
            copied = copy_parameters(
                parameters,
            )

        except Exception as exc:
            raise FederatedLearningError(
                "FederatedServer.initial_parameters "
                "could not be copied."
            ) from exc

        for client in clients.values():

            try:
                validate_parameters(
                    copied,
                    client.parameter_contract,
                )

            except FederatedLearningError as exc:
                raise FederatedLearningError(
                    "Initial global parameter validation "
                    "failed for "
                    f"client '{client.client_id}': "
                    f"{exc}"
                ) from exc

            except Exception as exc:
                raise FederatedLearningError(
                    "Unexpected initial global parameter "
                    "validation failure for "
                    f"client '{client.client_id}': "
                    f"{exc}"
                ) from exc

        return copied

    # ============================================================
    # Round-number validation
    # ============================================================

    @staticmethod
    def _validate_next_round_number(
        round_number: int,
        expected_round: int,
    ) -> None:
        """
        Validate strict one-based round progression.
        """

        if not isinstance(
            round_number,
            int,
        ):
            raise FederatedLearningError(
                "round_number must be an integer."
            )

        if isinstance(
            round_number,
            bool,
        ):
            raise FederatedLearningError(
                "round_number must be an integer, not bool."
            )

        if round_number < 1:
            raise FederatedLearningError(
                "round_number must be >= 1."
            )

        if round_number != expected_round:
            raise FederatedLearningError(
                "FederatedServer requires contiguous "
                "round progression: "
                f"expected round {expected_round}, "
                f"got {round_number}."
            )

    # ============================================================
    # Number-of-rounds validation
    # ============================================================

    @staticmethod
    def _validate_num_rounds(
        num_rounds: int,
    ) -> None:
        """
        Validate a positive number of additional rounds.
        """

        if not isinstance(
            num_rounds,
            int,
        ):
            raise FederatedLearningError(
                "num_rounds must be an integer."
            )

        if isinstance(
            num_rounds,
            bool,
        ):
            raise FederatedLearningError(
                "num_rounds must be an integer, not bool."
            )

        if num_rounds < 1:
            raise FederatedLearningError(
                "num_rounds must be >= 1."
            )

    # ============================================================
    # Round-result validation
    # ============================================================

    @staticmethod
    def _validate_round_execution(
        execution: RoundExecution,
        *,
        expected_round: int,
    ) -> None:
        """
        Validate coordinator output before committing server state.
        """

        if not isinstance(
            execution,
            RoundExecution,
        ):
            raise FederatedLearningError(
                "RoundCoordinator returned an invalid "
                "RoundExecution."
            )

        if (
            execution.result.round_number
            != expected_round
        ):
            raise FederatedLearningError(
                "RoundCoordinator returned a "
                "RoundExecution for unexpected round "
                f"{execution.result.round_number}; "
                f"expected {expected_round}."
            )

        if execution.result.status.value != "completed":
            raise FederatedLearningError(
                "FederatedServer can only commit a "
                "completed RoundExecution."
            )

        if not execution.aggregated_parameters:
            raise FederatedLearningError(
                "RoundCoordinator returned an empty "
                "aggregated parameter payload."
            )