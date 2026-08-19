"""
Federated round lifecycle and coordination for FedMed.

Phase 3.3-A establishes the round-level contract between the
future federated server, strategy, and existing Phase 3.2
FederatedClient.

Responsibilities
----------------
This module is responsible for:

- representing federated round lifecycle state
- recording structured client failures
- recording immutable round results
- carrying the result of one completed round
- coordinating one federated round through injected dependencies

This module intentionally does NOT:

- implement FedAvg mathematics
- implement any aggregation algorithm
- select clients according to a specific FL strategy
- implement local training
- implement local evaluation
- create datasets or DataLoaders
- modify model architecture
- implement networking or transport
- implement Flower-specific APIs
- own global server state
- implement checkpoint persistence

Architecture
------------

    FederatedServer
           |
           | current global parameters
           v
    RoundCoordinator
           |
           +------> Strategy
           |          |
           |          +---- client selection
           |          +---- aggregation policy
           |
           +------> FederatedClient
           |          |
           |          +---- fit()
           |          +---- evaluate()
           |
           v
      RoundExecution
           |
           +---- RoundResult
           |
           +---- aggregated parameters

The coordinator is reusable. The server owns global state and
invokes the coordinator once for each round.

The implementation remains framework-independent and uses the
existing FedMed parameter and FederatedClient contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Protocol, Sequence

import numpy as np

from src.common.exceptions import FederatedLearningError
from src.fl.client import (
    FederatedClient,
    FederatedEvaluateResult,
    FederatedFitResult,
)
from src.fl.parameters import (
    ParameterPayload,
    copy_parameters,
    validate_parameters,
)

if TYPE_CHECKING:
    from src.fl.parameters import ParameterContract


# ============================================================
# Internal strategy protocol
# ============================================================


class _RoundStrategyProtocol(Protocol):
    """
    Internal protocol describing the strategy operations required
    by RoundCoordinator.

    The public FederatedStrategy abstraction will be finalized in
    Phase 3.3-B.

    Keeping this protocol private prevents rounds.py from becoming
    the permanent owner of the public strategy API while allowing
    Phase 3.3-A to define the actual coordination boundary.
    """

    def select_fit_clients(
        self,
        clients: Sequence[FederatedClient],
        round_number: int,
    ) -> Sequence[FederatedClient]:
        ...

    def aggregate_fit(
        self,
        results: Mapping[str, FederatedFitResult],
        round_number: int,
    ) -> ParameterPayload:
        ...

    def select_evaluate_clients(
        self,
        clients: Sequence[FederatedClient],
        round_number: int,
    ) -> Sequence[FederatedClient]:
        ...


# ============================================================
# Round lifecycle
# ============================================================


class RoundState(Enum):
    """
    Lifecycle state of one federated learning round.

    A round progresses through the following normal lifecycle:

        CREATED
            |
            v
        SELECTING
            |
            v
        TRAINING
            |
            v
        AGGREGATING
            |
            +----------------+
            |                |
            v                v
        EVALUATING       COMPLETED
            |
            v
        COMPLETED

    A round may transition to FAILED from an active state when a
    fatal round-level condition occurs.

    Terminal states:
        COMPLETED
        FAILED
    """

    CREATED = "created"
    SELECTING = "selecting"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================
# Client failure
# ============================================================


@dataclass(frozen=True)
class ClientFailure:
    """
    Structured description of one client's failure during a round.

    Client failure is deliberately distinct from round failure.

    A client may fail while other clients successfully complete
    their work. The round may continue when the configured
    minimum-success policy is satisfied.

    Attributes
    ----------
    client_id:
        Stable federated client identifier.

    phase:
        Round phase during which the client failed.

        Valid client-failure phases are:
            TRAINING
            EVALUATING

    error:
        Human-readable description of the failure.
    """

    client_id: str
    phase: RoundState
    error: str

    def __post_init__(self) -> None:
        """Validate the failure record."""

        if not isinstance(self.client_id, str):
            raise FederatedLearningError(
                "ClientFailure.client_id must be a string, "
                f"got {type(self.client_id).__name__}."
            )

        if not self.client_id.strip():
            raise FederatedLearningError(
                "ClientFailure.client_id must be non-empty."
            )

        if not isinstance(self.phase, RoundState):
            raise FederatedLearningError(
                "ClientFailure.phase must be a RoundState, "
                f"got {type(self.phase).__name__}."
            )

        if self.phase not in {
            RoundState.TRAINING,
            RoundState.EVALUATING,
        }:
            raise FederatedLearningError(
                "ClientFailure.phase must be TRAINING or "
                f"EVALUATING, got {self.phase.value}."
            )

        if not isinstance(self.error, str):
            raise FederatedLearningError(
                "ClientFailure.error must be a string, "
                f"got {type(self.error).__name__}."
            )

        if not self.error.strip():
            raise FederatedLearningError(
                "ClientFailure.error must be non-empty."
            )


# ============================================================
# Round result
# ============================================================


@dataclass(frozen=True)
class RoundResult:
    """
    Immutable record describing the outcome of one federated round.

    The result contains round metadata and client-level outcomes.
    It deliberately does not own the server's global model state.

    Attributes
    ----------
    round_number:
        One-based federated round number.

    status:
        Final state of the round.

    selected_clients:
        Client IDs selected for the round's training operation.

    successful_clients:
        Client IDs whose training completed successfully.

    failed_clients:
        Structured records for clients that failed.

    fit_results:
        Successful local training results keyed by client ID.

    evaluation_results:
        Successful local evaluation results keyed by client ID.
    """

    round_number: int
    status: RoundState
    selected_clients: tuple[str, ...]
    successful_clients: tuple[str, ...]
    failed_clients: tuple[ClientFailure, ...]
    fit_results: Mapping[str, FederatedFitResult]
    evaluation_results: Mapping[str, FederatedEvaluateResult]

    def __post_init__(self) -> None:
        """Validate and freeze round result data."""

        if not isinstance(self.round_number, int):
            raise FederatedLearningError(
                "RoundResult.round_number must be an integer."
            )

        if isinstance(self.round_number, bool) or self.round_number < 1:
            raise FederatedLearningError(
                "RoundResult.round_number must be >= 1."
            )

        if not isinstance(self.status, RoundState):
            raise FederatedLearningError(
                "RoundResult.status must be a RoundState."
            )

        selected = tuple(self.selected_clients)
        successful = tuple(self.successful_clients)
        failures = tuple(self.failed_clients)

        self._validate_client_ids(
            selected,
            "selected_clients",
        )

        self._validate_client_ids(
            successful,
            "successful_clients",
        )

        if len(set(selected)) != len(selected):
            raise FederatedLearningError(
                "RoundResult.selected_clients contains duplicate IDs."
            )

        if len(set(successful)) != len(successful):
            raise FederatedLearningError(
                "RoundResult.successful_clients contains duplicate IDs."
            )

        selected_set = set(selected)
        successful_set = set(successful)

        if not successful_set.issubset(selected_set):
            raise FederatedLearningError(
                "Every successful client must be present in "
                "selected_clients."
            )

        failure_ids = [
            failure.client_id
            for failure in failures
        ]

        if len(set(failure_ids)) != len(failure_ids):
            raise FederatedLearningError(
                "RoundResult.failed_clients contains duplicate IDs."
            )

        failure_set = set(failure_ids)

        if not failure_set.issubset(selected_set):
            raise FederatedLearningError(
                "Every failed client must be present in "
                "selected_clients."
            )

        if successful_set.intersection(failure_set):
            raise FederatedLearningError(
                "A client cannot be both successful and failed."
            )

        fit_results = dict(self.fit_results)
        evaluation_results = dict(self.evaluation_results)

        if set(fit_results) != successful_set:
            raise FederatedLearningError(
                "fit_results keys must exactly match "
                "successful_clients."
            )

        evaluation_client_ids = set(evaluation_results)

        if not evaluation_client_ids.issubset(selected_set):
            raise FederatedLearningError(
                "evaluation_results may only contain selected clients."
            )

        object.__setattr__(
            self,
            "selected_clients",
            selected,
        )

        object.__setattr__(
            self,
            "successful_clients",
            successful,
        )

        object.__setattr__(
            self,
            "failed_clients",
            failures,
        )

        object.__setattr__(
            self,
            "fit_results",
            MappingProxyType(fit_results),
        )

        object.__setattr__(
            self,
            "evaluation_results",
            MappingProxyType(evaluation_results),
        )

    @staticmethod
    def _validate_client_ids(
        client_ids: tuple[str, ...],
        field_name: str,
    ) -> None:
        """Validate a tuple of client identifiers."""

        for client_id in client_ids:
            if not isinstance(client_id, str):
                raise FederatedLearningError(
                    f"{field_name} must contain only strings."
                )

            if not client_id.strip():
                raise FederatedLearningError(
                    f"{field_name} cannot contain empty client IDs."
                )


# ============================================================
# Round execution
# ============================================================


@dataclass(frozen=True)
class RoundExecution:
    """
    Operational output of one completed federated round.

    This object separates two concerns:

        RoundResult
            What happened during the round.

        aggregated_parameters
            The new global parameter payload produced by the
            strategy/aggregation layer.

    The server consumes aggregated_parameters and updates its
    global model state.

    Attributes
    ----------
    result:
        Immutable round outcome.

    aggregated_parameters:
        New global model parameters produced after successful
        aggregation.
    """

    result: RoundResult
    aggregated_parameters: ParameterPayload

    def __post_init__(self) -> None:
        """Validate and defensively copy the aggregated parameters."""

        if not isinstance(self.result, RoundResult):
            raise FederatedLearningError(
                "RoundExecution.result must be a RoundResult."
            )

        copied = copy_parameters(
            self.aggregated_parameters,
        )

        object.__setattr__(
            self,
            "aggregated_parameters",
            copied,
        )


# ============================================================
# Round coordinator
# ============================================================


class RoundCoordinator:
    """
    Reusable coordinator for executing one federated round.

    The coordinator is deliberately stateless with respect to
    federation progress. FederatedServer owns:

        - global parameters
        - current round number
        - round history
        - long-lived federation state

    RoundCoordinator owns the lifecycle of an individual round.

    Dependencies
    ------------
    strategy:
        Strategy implementation responsible for client selection
        and aggregation policy.

    clients:
        Registered federated clients keyed by stable client ID.

    The coordinator does not directly implement a specific
    aggregation algorithm.

    Notes
    -----
    The public FederatedStrategy contract is finalized in
    Phase 3.3-B. This module therefore uses a private structural
    protocol so the round implementation can be developed without
    coupling it to a concrete strategy implementation.
    """

    def __init__(
        self,
        strategy: _RoundStrategyProtocol,
        clients: Mapping[str, FederatedClient],
    ) -> None:
        """
        Construct a reusable round coordinator.

        Parameters
        ----------
        strategy:
            Strategy object providing client-selection and
            aggregation operations.

        clients:
            Mapping of stable client IDs to FederatedClient
            instances.
        """

        if strategy is None:
            raise FederatedLearningError(
                "RoundCoordinator requires a strategy."
            )

        if not isinstance(clients, Mapping):
            raise FederatedLearningError(
                "RoundCoordinator.clients must be a mapping."
            )

        normalized_clients = dict(clients)

        self._validate_clients(normalized_clients)

        self._strategy = strategy
        self._clients = MappingProxyType(normalized_clients)

    # ========================================================
    # Public properties
    # ========================================================

    @property
    def clients(self) -> Mapping[str, FederatedClient]:
        """Return the registered clients as a read-only mapping."""

        return self._clients

    @property
    def strategy(self) -> _RoundStrategyProtocol:
        """Return the configured federation strategy."""

        return self._strategy

    # ========================================================
    # Round execution
    # ========================================================

    def execute_round(
        self,
        round_number: int,
        parameters: ParameterPayload,
        *,
        evaluate: bool = False,
    ) -> RoundExecution:
        """
        Execute one federated round.

        Lifecycle
        ---------
            CREATED
                ↓
            SELECTING
                ↓
            TRAINING
                ↓
            AGGREGATING
                ↓
            [EVALUATING]
                ↓
            COMPLETED

        Expected client-level federated failures are recorded as
        ClientFailure objects. The strategy is responsible for
        determining how many successful updates are required for
        aggregation.

        Parameters
        ----------
        round_number:
            One-based federated round number.

        parameters:
            Current global parameter payload.

        evaluate:
            Whether local client evaluation should be executed after
            successful aggregation.

            Evaluation policy will be finalized in Phase 3.3-B.
            This explicit execution flag keeps the round contract
            usable without embedding a policy into rounds.py.

        Returns
        -------
        RoundExecution
            Round result plus the newly aggregated global parameters.

        Raises
        ------
        FederatedLearningError
            If a fatal round-level condition occurs.
        """

        self._validate_round_number(round_number)

        self._validate_global_parameters(parameters)

        state = RoundState.CREATED

        selected_clients: tuple[str, ...] = ()
        successful_clients: tuple[str, ...] = ()
        failures: list[ClientFailure] = []
        fit_results: dict[str, FederatedFitResult] = {}
        evaluation_results: dict[
            str,
            FederatedEvaluateResult,
        ] = {}

        try:
            # ------------------------------------------------
            # Client selection
            # ------------------------------------------------

            state = RoundState.SELECTING

            selected = self._strategy.select_fit_clients(
                tuple(self._clients.values()),
                round_number,
            )

            selected_clients = self._normalize_selected_clients(
                selected,
            )

            self._validate_selection(
                selected_clients,
            )

            # ------------------------------------------------
            # Local training
            # ------------------------------------------------

            state = RoundState.TRAINING

            for client_id in selected_clients:
                client = self._clients[client_id]

                try:
                    result = client.fit(parameters)

                except FederatedLearningError as exc:
                    failures.append(
                        ClientFailure(
                            client_id=client_id,
                            phase=RoundState.TRAINING,
                            error=str(exc),
                        )
                    )
                    continue

                except Exception as exc:
                    raise FederatedLearningError(
                        f"Unexpected failure while executing "
                        f"training for client '{client_id}': {exc}"
                    ) from exc

                if not isinstance(result, FederatedFitResult):
                    raise FederatedLearningError(
                        f"Client '{client_id}' returned an invalid "
                        "FederatedFitResult."
                    )

                fit_results[client_id] = result

            successful_clients = tuple(fit_results)

            # ------------------------------------------------
            # Minimum participation
            # ------------------------------------------------

            if not successful_clients:
                raise FederatedLearningError(
                    f"Round {round_number} produced no successful "
                    "client training results."
                )

            # ------------------------------------------------
            # Aggregation
            # ------------------------------------------------

            state = RoundState.AGGREGATING

            aggregated_parameters = (
                self._strategy.aggregate_fit(
                    fit_results,
                    round_number,
                )
            )

            self._validate_aggregated_parameters(
                aggregated_parameters,
                successful_clients,
            )

            # ------------------------------------------------
            # Optional evaluation
            # ------------------------------------------------

            if evaluate:
                state = RoundState.EVALUATING

                evaluation_clients = (
                    self._strategy.select_evaluate_clients(
                        tuple(self._clients.values()),
                        round_number,
                    )
                )

                evaluation_ids = (
                    self._normalize_selected_clients(
                        evaluation_clients,
                    )
                )

                for client_id in evaluation_ids:
                    client = self._clients[client_id]

                    try:
                        result = client.evaluate(
                            aggregated_parameters,
                        )

                    except FederatedLearningError as exc:
                        failures.append(
                            ClientFailure(
                                client_id=client_id,
                                phase=RoundState.EVALUATING,
                                error=str(exc),
                            )
                        )
                        continue

                    except Exception as exc:
                        raise FederatedLearningError(
                            f"Unexpected failure while executing "
                            f"evaluation for client "
                            f"'{client_id}': {exc}"
                        ) from exc

                    if not isinstance(
                        result,
                        FederatedEvaluateResult,
                    ):
                        raise FederatedLearningError(
                            f"Client '{client_id}' returned an invalid "
                            "FederatedEvaluateResult."
                        )

                    evaluation_results[client_id] = result

            # ------------------------------------------------
            # Complete
            # ------------------------------------------------

            state = RoundState.COMPLETED

            result = RoundResult(
                round_number=round_number,
                status=state,
                selected_clients=selected_clients,
                successful_clients=successful_clients,
                failed_clients=tuple(failures),
                fit_results=fit_results,
                evaluation_results=evaluation_results,
            )

            return RoundExecution(
                result=result,
                aggregated_parameters=aggregated_parameters,
            )

        except FederatedLearningError:
            # The coordinator intentionally does not manufacture a
            # RoundResult for a fatal exception. The server/failure
            # policy layer will decide whether to retry, abort, or
            # persist the failed-round state.
            raise

        except Exception as exc:
            raise FederatedLearningError(
                f"Round {round_number} failed during "
                f"{state.value}: {exc}"
            ) from exc

    # ========================================================
    # Validation helpers
    # ========================================================

    @staticmethod
    def _validate_clients(
        clients: Mapping[str, FederatedClient],
    ) -> None:
        """Validate the registered client mapping."""

        if not clients:
            raise FederatedLearningError(
                "RoundCoordinator requires at least one client."
            )

        for client_id, client in clients.items():
            if not isinstance(client_id, str):
                raise FederatedLearningError(
                    "Client registry keys must be strings."
                )

            if not client_id.strip():
                raise FederatedLearningError(
                    "Client registry contains an empty client ID."
                )

            if not isinstance(client, FederatedClient):
                raise FederatedLearningError(
                    f"Client '{client_id}' must be a FederatedClient, "
                    f"got {type(client).__name__}."
                )

            if client.client_id != client_id:
                raise FederatedLearningError(
                    "Client registry key does not match the client's "
                    f"own ID: key='{client_id}', "
                    f"client_id='{client.client_id}'."
                )

    @staticmethod
    def _validate_round_number(
        round_number: int,
    ) -> None:
        """Validate the one-based round number."""

        if not isinstance(round_number, int):
            raise FederatedLearningError(
                "round_number must be an integer."
            )

        if isinstance(round_number, bool) or round_number < 1:
            raise FederatedLearningError(
                "round_number must be >= 1."
            )

    def _validate_global_parameters(
        self,
        parameters: ParameterPayload,
    ) -> None:
        """
        Validate the incoming global parameters against every
        registered client's parameter contract.

        This ensures that the server cannot start a round with a
        parameter payload incompatible with one of its clients.

        Detailed validation remains centralized in Phase 3.1.
        """

        for client in self._clients.values():
            try:
                validate_parameters(
                    parameters,
                    client.parameter_contract,
                )
            except Exception as exc:
                if isinstance(exc, FederatedLearningError):
                    raise FederatedLearningError(
                        "Global parameter validation failed for "
                        f"client '{client.client_id}': {exc}"
                    ) from exc

                raise FederatedLearningError(
                    "Unexpected global parameter validation failure "
                    f"for client '{client.client_id}': {exc}"
                ) from exc

    def _validate_aggregated_parameters(
        self,
        parameters: ParameterPayload,
        successful_clients: Sequence[str],
    ) -> None:
        """
        Validate the aggregated parameter payload against the
        successful clients' parameter contracts.

        Aggregation itself belongs to the Aggregator/Strategy layer.
        This method only verifies the returned payload.
        """

        if not successful_clients:
            raise FederatedLearningError(
                "Cannot validate aggregated parameters without "
                "successful clients."
            )

        for client_id in successful_clients:
            client = self._clients[client_id]

            try:
                validate_parameters(
                    parameters,
                    client.parameter_contract,
                )
            except Exception as exc:
                if isinstance(exc, FederatedLearningError):
                    raise FederatedLearningError(
                        "Aggregated parameter validation failed for "
                        f"client '{client_id}': {exc}"
                    ) from exc

                raise FederatedLearningError(
                    "Unexpected aggregated parameter validation "
                    f"failure for client '{client_id}': {exc}"
                ) from exc

    def _validate_selection(
        self,
        selected_clients: tuple[str, ...],
    ) -> None:
        """Validate that strategy selection references registered clients."""

        if not selected_clients:
            raise FederatedLearningError(
                "Strategy selected no clients for training."
            )

        unknown_clients = [
            client_id
            for client_id in selected_clients
            if client_id not in self._clients
        ]

        if unknown_clients:
            raise FederatedLearningError(
                "Strategy selected unknown clients: "
                + ", ".join(unknown_clients)
            )

    def _normalize_selected_clients(
        self,
        clients: Sequence[FederatedClient],
    ) -> tuple[str, ...]:
        """
        Convert selected client objects into stable client IDs.

        The strategy returns actual FederatedClient objects so the
        coordinator retains the existing client abstraction.
        """

        if not isinstance(clients, Sequence):
            raise FederatedLearningError(
                "Strategy client selection must return a sequence "
                "of FederatedClient objects."
            )

        client_ids: list[str] = []

        for client in clients:
            if not isinstance(client, FederatedClient):
                raise FederatedLearningError(
                    "Strategy selection must contain only "
                    "FederatedClient objects, got "
                    f"{type(client).__name__}."
                )

            client_ids.append(client.client_id)

        if len(set(client_ids)) != len(client_ids):
            raise FederatedLearningError(
                "Strategy selected duplicate clients."
            )

        return tuple(client_ids)