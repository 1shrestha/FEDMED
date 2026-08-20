"""
Federated strategy abstractions for FedMed.

Phase 3.3-B
-----------

This module defines the federation-policy layer between
RoundCoordinator and the concrete parameter aggregation layer.

Architecture
------------

                    RoundCoordinator
                           |
                           v
                  FederatedStrategy
                           |
              +------------+------------+
              |                         |
              v                         v
      Client selection          aggregate_fit()
                                        |
                                        v
                                   Aggregator
                                        |
                                        v
                                FedAvgAggregator


Responsibilities
----------------

FederatedStrategy is responsible for:

- defining the public strategy contract
- selecting clients for local training
- selecting clients for local evaluation
- defining the federation policy boundary
- delegating parameter aggregation

FedAvgStrategy provides the first concrete FedMed strategy.

Its initial baseline policy is intentionally deterministic:

- select all available clients for training
- select all available clients for evaluation
- delegate aggregation to the configured Aggregator


This module intentionally does NOT:

- implement FedAvg mathematics
- average NumPy arrays
- calculate sample weights
- implement local training
- implement local evaluation
- create datasets
- create DataLoaders
- manage global model state
- execute federated rounds
- implement networking
- implement Flower APIs
- aggregate metrics


Compatibility
-------------

The public methods defined here intentionally match the private
_round strategy protocol currently used by src/fl/rounds.py:

    select_fit_clients(...)
    aggregate_fit(...)
    select_evaluate_clients(...)

The RoundCoordinator therefore does not need to be redesigned
for Phase 3.3-B.


Design principle
----------------

Strategy answers:

    "How should this federation operate?"

Aggregator answers:

    "How should successful client model parameters be
     mathematically combined?"

The Strategy therefore delegates aggregation instead of
implementing it.

This preserves the separation:

    federation policy
            !=
    mathematical aggregation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from src.common.exceptions import FederatedLearningError
from src.fl.aggregation import Aggregator
from src.fl.client import (
    FederatedClient,
    FederatedFitResult,
)
from src.fl.parameters import ParameterPayload


# ============================================================
# Federated Strategy
# ============================================================


class FederatedStrategy(ABC):
    """
    Abstract federation strategy for FedMed.

    A FederatedStrategy defines policy for one federated round.

    The strategy is intentionally separated from the round
    coordinator.

    RoundCoordinator owns:

        - executing the round lifecycle
        - invoking clients
        - recording failures
        - constructing RoundResult
        - carrying aggregated parameters

    FederatedStrategy owns:

        - deciding which clients participate
        - defining aggregation policy
        - selecting evaluation participants

    Concrete strategies may implement different federation
    policies without changing RoundCoordinator.
    """

    @abstractmethod
    def select_fit_clients(
        self,
        clients: Sequence[FederatedClient],
        round_number: int,
    ) -> Sequence[FederatedClient]:
        """
        Select clients for local training.

        Parameters
        ----------
        clients:
            Available federated clients.

        round_number:
            One-based federated round number.

        Returns
        -------
        Sequence[FederatedClient]
            Clients selected for local training.

        Notes
        -----
        The returned sequence must contain only clients supplied
        by the coordinator.

        Concrete strategies determine the selection policy.

        The initial FedAvgStrategy selects all available clients
        deterministically.
        """
        raise NotImplementedError

    @abstractmethod
    def aggregate_fit(
        self,
        results: Mapping[str, FederatedFitResult],
        round_number: int,
    ) -> ParameterPayload:
        """
        Aggregate successful local training results.

        Parameters
        ----------
        results:
            Successful local training results keyed by stable
            client ID.

        round_number:
            One-based federated round number.

        Returns
        -------
        ParameterPayload
            New global model parameters.

        Notes
        -----
        This method defines the Strategy-level aggregation
        boundary.

        Concrete strategies should delegate mathematical
        aggregation to an Aggregator rather than implementing
        parameter arithmetic directly.
        """
        raise NotImplementedError

    @abstractmethod
    def select_evaluate_clients(
        self,
        clients: Sequence[FederatedClient],
        round_number: int,
    ) -> Sequence[FederatedClient]:
        """
        Select clients for local evaluation.

        Parameters
        ----------
        clients:
            Available federated clients.

        round_number:
            One-based federated round number.

        Returns
        -------
        Sequence[FederatedClient]
            Clients selected for local evaluation.

        Notes
        -----
        Evaluation selection is intentionally separate from
        training selection.

        A future strategy may use different policies for fitting
        and evaluation.
        """
        raise NotImplementedError

    # ========================================================
    # Shared validation
    # ========================================================

    @staticmethod
    def _validate_round_number(
        round_number: int,
    ) -> None:
        """
        Validate a federated round number.

        FedMed uses one-based round numbering.
        """

        if isinstance(
            round_number,
            bool,
        ) or not isinstance(
            round_number,
            int,
        ):
            raise FederatedLearningError(
                "round_number must be an integer, "
                f"got {type(round_number).__name__}."
            )

        if round_number < 1:
            raise FederatedLearningError(
                "round_number must be >= 1, "
                f"got {round_number}."
            )

    @staticmethod
    def _validate_clients(
        clients: Sequence[FederatedClient],
    ) -> tuple[FederatedClient, ...]:
        """
        Validate and normalize an available-client sequence.

        A tuple is returned so strategy implementations cannot
        accidentally mutate the coordinator's client collection.
        """

        if isinstance(
            clients,
            (str, bytes),
        ):
            raise FederatedLearningError(
                "clients must be a sequence of "
                "FederatedClient objects."
            )

        if not isinstance(
            clients,
            Sequence,
        ):
            raise FederatedLearningError(
                "clients must be a sequence of "
                f"FederatedClient objects, got "
                f"{type(clients).__name__}."
            )

        normalized = tuple(clients)

        if not normalized:
            raise FederatedLearningError(
                "Strategy requires at least one available "
                "FederatedClient."
            )

        seen_ids: set[str] = set()

        for client in normalized:
            if not isinstance(
                client,
                FederatedClient,
            ):
                raise FederatedLearningError(
                    "Strategy clients must contain only "
                    "FederatedClient objects, got "
                    f"{type(client).__name__}."
                )

            client_id = client.client_id

            if not isinstance(
                client_id,
                str,
            ) or not client_id.strip():
                raise FederatedLearningError(
                    "Every FederatedClient must have a "
                    "non-empty string client_id."
                )

            if client_id in seen_ids:
                raise FederatedLearningError(
                    "Strategy received duplicate client ID: "
                    f"'{client_id}'."
                )

            seen_ids.add(client_id)

        return normalized

    @staticmethod
    def _validate_selected_clients(
        selected_clients: Sequence[FederatedClient],
        available_clients: Sequence[FederatedClient],
    ) -> tuple[FederatedClient, ...]:
        """
        Validate a strategy's selected clients.

        Selection must:

        - contain FederatedClient instances
        - contain no duplicates
        - reference only available clients

        The returned tuple is an immutable snapshot of the
        selection.
        """

        if isinstance(
            selected_clients,
            (str, bytes),
        ):
            raise FederatedLearningError(
                "Selected clients must be a sequence of "
                "FederatedClient objects."
            )

        if not isinstance(
            selected_clients,
            Sequence,
        ):
            raise FederatedLearningError(
                "Selected clients must be a sequence, "
                f"got {type(selected_clients).__name__}."
            )

        selected = tuple(
            selected_clients,
        )

        available_ids = {
            client.client_id
            for client in available_clients
        }

        selected_ids: set[str] = set()

        for client in selected:
            if not isinstance(
                client,
                FederatedClient,
            ):
                raise FederatedLearningError(
                    "Strategy selection must contain only "
                    "FederatedClient objects, got "
                    f"{type(client).__name__}."
                )

            client_id = client.client_id

            if client_id in selected_ids:
                raise FederatedLearningError(
                    "Strategy selected duplicate client: "
                    f"'{client_id}'."
                )

            if client_id not in available_ids:
                raise FederatedLearningError(
                    "Strategy selected client "
                    f"'{client_id}' that is not present "
                    "in the available client set."
                )

            selected_ids.add(client_id)

        return selected

    @staticmethod
    def _validate_fit_results(
        results: Mapping[str, FederatedFitResult],
    ) -> Mapping[str, FederatedFitResult]:
        """
        Validate the Strategy-level aggregation input.

        Aggregator performs algorithm-specific validation.

        Strategy validates only the contract necessary to safely
        cross the Strategy -> Aggregator boundary.
        """

        if not isinstance(
            results,
            Mapping,
        ):
            raise FederatedLearningError(
                "aggregate_fit() results must be a mapping "
                "from client IDs to FederatedFitResult objects, "
                f"got {type(results).__name__}."
            )

        if not results:
            raise FederatedLearningError(
                "aggregate_fit() cannot aggregate an empty "
                "set of client results."
            )

        for client_id, result in results.items():
            if not isinstance(
                client_id,
                str,
            ):
                raise FederatedLearningError(
                    "aggregate_fit() result keys must be "
                    "strings, got "
                    f"{type(client_id).__name__}."
                )

            if not client_id.strip():
                raise FederatedLearningError(
                    "aggregate_fit() cannot contain an empty "
                    "client ID."
                )

            if not isinstance(
                result,
                FederatedFitResult,
            ):
                raise FederatedLearningError(
                    "aggregate_fit() result values must be "
                    "FederatedFitResult instances, got "
                    f"{type(result).__name__} for client "
                    f"'{client_id}'."
                )

        return results


# ============================================================
# FedAvg Strategy
# ============================================================


class FedAvgStrategy(FederatedStrategy):
    """
    Baseline FedAvg federation strategy.

    The strategy deliberately separates federation policy from
    mathematical aggregation.

    Policy
    ------

    Training selection:

        select all available clients

    Evaluation selection:

        select all available clients

    Parameter aggregation:

        delegate to the configured Aggregator

    By default, the strategy creates a FedAvgAggregator through
    the aggregation module. The dependency is injected through
    the Aggregator abstraction so the strategy can later be used
    with a different aggregation implementation without changing
    its public contract.

    Example
    -------

        strategy = FedAvgStrategy()

    or:

        strategy = FedAvgStrategy(
            aggregator=custom_aggregator,
        )

    The second form is useful for future research experiments.

    Important
    ---------

    This class does NOT implement FedAvg mathematics.

    In particular, it does not:

        - calculate sample weights
        - average NumPy arrays
        - inspect parameter tensors
        - modify client results

    All mathematical aggregation is delegated to Aggregator.
    """

    def __init__(
        self,
        aggregator: Aggregator,
    ) -> None:
        """
        Construct a FedAvg strategy.

        Parameters
        ----------
        aggregator:
            Aggregator implementation responsible for the actual
            parameter aggregation.

        Raises
        ------
        FederatedLearningError
            If aggregator is not an Aggregator instance.
        """

        if not isinstance(
            aggregator,
            Aggregator,
        ):
            raise FederatedLearningError(
                "FedAvgStrategy requires an Aggregator "
                f"instance, got {type(aggregator).__name__}."
            )

        self._aggregator = aggregator

    # ========================================================
    # Public properties
    # ========================================================

    @property
    def aggregator(self) -> Aggregator:
        """
        Return the configured aggregation component.

        The Aggregator is intentionally exposed read-only so tests,
        diagnostics, and future runtime wiring can inspect the
        strategy dependency without allowing accidental
        replacement through the property.
        """

        return self._aggregator

    # ========================================================
    # Client selection
    # ========================================================

    def select_fit_clients(
        self,
        clients: Sequence[FederatedClient],
        round_number: int,
    ) -> Sequence[FederatedClient]:
        """
        Select clients for local training.

        Baseline FedAvg policy:

            select all available clients

        Selection is deterministic and preserves the order supplied
        by RoundCoordinator.

        Parameters
        ----------
        clients:
            Available federated clients.

        round_number:
            Current one-based round number.

        Returns
        -------
        tuple[FederatedClient, ...]
            All available clients in their existing order.

        Notes
        -----
        The round number is validated even though the baseline
        all-client policy does not currently use it.

        This preserves a stable Strategy contract for future
        round-dependent selection policies.
        """

        self._validate_round_number(
            round_number,
        )

        available = self._validate_clients(
            clients,
        )

        selected = tuple(
            available,
        )

        return self._validate_selected_clients(
            selected,
            available,
        )

    def select_evaluate_clients(
        self,
        clients: Sequence[FederatedClient],
        round_number: int,
    ) -> Sequence[FederatedClient]:
        """
        Select clients for local evaluation.

        Baseline FedAvg policy:

            select all available clients

        Selection is deterministic and preserves the order supplied
        by RoundCoordinator.

        Evaluation selection is deliberately separate from training
        selection so future strategies can independently control
        the two populations.
        """

        self._validate_round_number(
            round_number,
        )

        available = self._validate_clients(
            clients,
        )

        selected = tuple(
            available,
        )

        return self._validate_selected_clients(
            selected,
            available,
        )

    # ========================================================
    # Aggregation delegation
    # ========================================================

    def aggregate_fit(
        self,
        results: Mapping[str, FederatedFitResult],
        round_number: int,
    ) -> ParameterPayload:
        """
        Delegate successful fit-result aggregation to Aggregator.

        This method exists because RoundCoordinator already defines
        the Strategy-level contract:

            Strategy.aggregate_fit(results, round_number)

        The Strategy does NOT implement parameter aggregation.

        Lifecycle
        ---------

            RoundCoordinator
                    |
                    v
            FedAvgStrategy.aggregate_fit()
                    |
                    v
            Aggregator.aggregate()
                    |
                    v
            ParameterPayload

        Parameters
        ----------
        results:
            Successful local training results keyed by client ID.

        round_number:
            Current one-based round number.

        Returns
        -------
        ParameterPayload
            Aggregated global model parameters.

        Raises
        ------
        FederatedLearningError
            If the input is invalid or the configured Aggregator
            fails.
        """

        self._validate_round_number(
            round_number,
        )

        validated_results = (
            self._validate_fit_results(
                results,
            )
        )

        try:
            aggregated_parameters = (
                self._aggregator.aggregate(
                    validated_results,
                )
            )

        except FederatedLearningError:
            # Preserve FedMed's existing domain exception without
            # wrapping it unnecessarily.
            raise

        except Exception as exc:
            raise FederatedLearningError(
                "Strategy aggregation failed during "
                f"round {round_number}: {exc}"
            ) from exc

        if not isinstance(
            aggregated_parameters,
            list,
        ):
            raise FederatedLearningError(
                "Aggregator returned an invalid parameter "
                "payload. Expected ParameterPayload "
                "(list[np.ndarray]), got "
                f"{type(aggregated_parameters).__name__}."
            )

        return aggregated_parameters