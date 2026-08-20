"""
Generic aggregation contract for FedMed.

Phase 3.3-B defines the abstraction boundary between FedMed's
federation strategy layer and concrete parameter aggregation
algorithms.

Responsibilities
----------------
This module is responsible for:

- defining the generic Aggregator abstraction
- defining the canonical aggregation input contract
- defining the canonical aggregation output contract
- validating common aggregation-input invariants
- remaining independent of any specific aggregation algorithm

The concrete mathematical aggregation algorithm does NOT belong
in this module.

For example:

    Aggregator
        |
        +---- FedAvgAggregator
        +---- future aggregation algorithms

The first concrete implementation is FedAvgAggregator, located in:

    src/aggregation/fedavg.py


Aggregation contract
--------------------

    Mapping[str, FederatedFitResult]
                    |
                    v
             Aggregator.aggregate()
                    |
                    v
             ParameterPayload


The aggregation input uses FederatedFitResult from Phase 3.2.

The aggregation output uses ParameterPayload from Phase 3.1.

Therefore this module intentionally builds on the existing
FedMed contracts rather than introducing new parameter/result
representations.


Responsibilities deliberately excluded
--------------------------------------

This module does NOT:

- implement FedAvg
- implement parameter averaging
- calculate sample weights
- aggregate metrics
- select clients
- manage federated rounds
- execute local training
- execute evaluation
- create datasets or DataLoaders
- manage global server state
- implement networking
- import Flower-specific APIs
- depend on a specific model architecture


Design rationale
----------------

FedMed deliberately separates federation policy from mathematical
aggregation.

Strategy is responsible for:

    - client selection
    - federation policy
    - round-level decisions
    - delegating aggregation

Aggregator is responsible for:

    - combining successful client results into global parameters

This separation allows a strategy to use different aggregation
algorithms without changing the round coordinator or federated
client.

The abstraction is intentionally narrow because Phase 3.3-B first
establishes vanilla FedAvg as the baseline. More advanced
algorithms can introduce additional contracts later when their
mathematical/state requirements are actually known.


Framework independence
----------------------

FedMed's internal representation remains:

    ParameterPayload = list[np.ndarray]

Framework-specific conversions, including Flower parameter
representations, belong at the outer runtime boundary and are not
part of this abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Final

from src.common.exceptions import FederatedLearningError
from src.fl.client import FederatedFitResult
from src.fl.parameters import ParameterPayload


# ============================================================
# Module constants
# ============================================================

AGGREGATION_INPUT_TYPE_NAME: Final[str] = (
    "Mapping[str, FederatedFitResult]"
)


# ============================================================
# Aggregator
# ============================================================


class Aggregator(ABC):
    """
    Abstract contract for federated parameter aggregation.

    An Aggregator receives successful local training results from
    participating federated clients and produces the next global
    parameter payload.

    The Aggregator is intentionally independent of:

        - Strategy
        - RoundCoordinator
        - FederatedClient execution
        - Trainer
        - Evaluator
        - Dataset/DataLoader
        - server state
        - network transport
        - Flower APIs

    Concrete subclasses implement the mathematical aggregation
    algorithm.

    Example:

        results = {
            "client_1": fit_result_1,
            "client_2": fit_result_2,
        }

        parameters = aggregator.aggregate(results)

    Contract
    --------
    Input:

        Mapping[str, FederatedFitResult]

    Output:

        ParameterPayload

    The returned ParameterPayload must be a new parameter payload
    owned by the aggregation implementation. Aggregators must not
    mutate parameter arrays contained in the supplied results.

    Notes
    -----
    Phase 3.3-B uses successful training results only.

    Client failures are handled by RoundCoordinator before the
    aggregation boundary is reached. This keeps failure policy
    outside the mathematical aggregation component.
    """

    @abstractmethod
    def aggregate(
        self,
        results: Mapping[str, FederatedFitResult],
    ) -> ParameterPayload:
        """
        Aggregate successful federated client results.

        Parameters
        ----------
        results:
            Mapping from stable federated client IDs to successful
            local training results.

        Returns
        -------
        ParameterPayload
            Newly constructed global model parameter payload.

        Raises
        ------
        FederatedLearningError
            If the aggregation input violates the common
            Aggregator contract or if the concrete aggregation
            algorithm cannot produce a valid result.

        Notes
        -----
        Concrete implementations are responsible for their
        algorithm-specific validation and mathematical behavior.

        For example, FedAvgAggregator additionally validates:

            - positive sample counts
            - compatible parameter layouts
            - compatible dtypes
            - floating-point aggregation requirements
            - non-floating state consistency

        The generic Aggregator contract does not encode those
        FedAvg-specific rules.
        """
        raise NotImplementedError

    # ========================================================
    # Common input validation
    # ========================================================

    @staticmethod
    def _validate_results(
        results: Mapping[str, FederatedFitResult],
    ) -> None:
        """
        Validate invariants common to all FedMed aggregators.

        This method intentionally performs only validation that is
        meaningful for every aggregation algorithm supported by the
        current FedMed contract.

        Algorithm-specific validation belongs to the concrete
        Aggregator implementation.

        Common invariants
        -----------------
        1. Results must be a Mapping.
        2. The mapping must not be empty.
        3. Every key must be a non-empty string.
        4. Every value must be a FederatedFitResult.
        5. Mapping keys must correspond to unique client IDs.

        Parameters
        ----------
        results:
            Candidate aggregation results.

        Raises
        ------
        FederatedLearningError
            If the common aggregation contract is violated.
        """

        if not isinstance(results, Mapping):
            raise FederatedLearningError(
                "Aggregator results must be a "
                f"{AGGREGATION_INPUT_TYPE_NAME}, "
                f"got {type(results).__name__}."
            )

        if not results:
            raise FederatedLearningError(
                "Aggregator cannot aggregate an empty set "
                "of client results."
            )

        for client_id, result in results.items():
            if not isinstance(client_id, str):
                raise FederatedLearningError(
                    "Aggregator result keys must be strings, "
                    f"got {type(client_id).__name__}."
                )

            if not client_id.strip():
                raise FederatedLearningError(
                    "Aggregator result contains an empty "
                    "client ID."
                )

            if not isinstance(
                result,
                FederatedFitResult,
            ):
                raise FederatedLearningError(
                    "Aggregator result values must be "
                    "FederatedFitResult instances, "
                    f"got {type(result).__name__} for "
                    f"client '{client_id}'."
                )

    # ========================================================
    # Representation
    # ========================================================

    def __repr__(self) -> str:
        """
        Return a concise representation of the Aggregator.

        Concrete implementations inherit this representation
        unless they explicitly provide their own.
        """

        return f"{self.__class__.__name__}()"