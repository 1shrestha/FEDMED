"""
FedAvg parameter aggregation for FedMed.

Phase 3.3-B
-----------

This module implements the first concrete FedMed aggregation
algorithm: Federated Averaging (FedAvg).

Architecture
------------

    Mapping[str, FederatedFitResult]
                    |
                    v
             FedAvgAggregator
                    |
                    v
             ParameterPayload


FedAvg
------

For successful clients k:

                    n_k
    w_k = -------------------------
                 N

    N = sum(n_k)

The aggregated parameter for each model-state entry is:

                     K
    W = sum (w_k * W_k)
                    k=1


where:

    W_k = client k's locally trained parameter array
    n_k = number of local training examples for client k
    N   = total number of training examples

This is the standard sample-count-weighted FedAvg formulation.


Responsibilities
----------------

This module is responsible for:

- implementing FedAvg parameter aggregation
- validating aggregation-specific input invariants
- computing sample-count-based client weights
- aggregating floating-point model state
- preserving compatible non-floating model state
- returning an independent ParameterPayload


This module intentionally does NOT:

- select clients
- manage rounds
- execute local training
- execute evaluation
- aggregate metrics
- manage server/global model state
- implement networking
- implement Flower APIs
- implement Strategy policy
- create datasets or DataLoaders


Parameter representation
------------------------

FedMed's canonical internal parameter representation is:

    ParameterPayload = list[np.ndarray]

The aggregator therefore operates directly on the existing
framework-independent FedMed parameter representation.

No PyTorch state_dict or Flower Parameters object is introduced
here.


Non-floating model state
------------------------

FedMed's Phase 3.1 parameter contract represents the complete
model state, not only trainable floating-point tensors.

Some model states may therefore contain non-floating arrays,
for example integer buffers.

Non-floating values are NOT mathematically averaged.

Instead:

    - if all participating clients have identical values,
      the value is preserved;

    - if participating clients disagree, aggregation fails.

This prevents meaningless operations such as averaging integer
state and producing an arbitrary value.


Input immutability
------------------

Aggregation never modifies:

    FederatedFitResult.parameters

or any NumPy array supplied by a client.

The returned ParameterPayload is newly allocated and independent
from all client inputs.


Framework independence
----------------------

This module depends only on FedMed's internal abstractions.

Framework-specific parameter conversion belongs outside the
FedMed aggregation layer.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from src.common.exceptions import FederatedLearningError
from src.fl.aggregation import Aggregator
from src.fl.client import FederatedFitResult
from src.fl.parameters import ParameterPayload


# ============================================================
# FedAvg Aggregator
# ============================================================


class FedAvgAggregator(Aggregator):
    """
    Sample-count-weighted Federated Averaging aggregator.

    FedAvg computes the next global model by taking a weighted
    average of the successfully trained client models.

    The weight of client k is:

        n_k
        -----
          N

    where:

        n_k = client k's number of training examples
        N   = total number of training examples

    Example
    -------

        Client A:
            parameters = WA
            num_examples = 100

        Client B:
            parameters = WB
            num_examples = 300

        Total:
            N = 400

        Weights:
            A = 0.25
            B = 0.75

        Global:
            W = 0.25 * WA + 0.75 * WB

    The implementation is intentionally stateless.

    A FedAvgAggregator instance can therefore be safely reused
    across multiple rounds.
    """

    # ========================================================
    # Public API
    # ========================================================

    def aggregate(
        self,
        results: Mapping[str, FederatedFitResult],
    ) -> ParameterPayload:
        """
        Aggregate successful client model parameters using FedAvg.

        Parameters
        ----------
        results:
            Mapping from stable federated client IDs to successful
            FederatedFitResult objects.

        Returns
        -------
        ParameterPayload
            Newly allocated global model parameters.

        Raises
        ------
        FederatedLearningError
            If:

            - results is invalid or empty
            - a client result is invalid
            - num_examples is invalid
            - parameter counts differ
            - parameter shapes differ
            - parameter dtypes differ
            - parameter values contain non-finite values
            - floating-point parameter aggregation is impossible
            - non-floating state differs between clients

        Notes
        -----
        Client failures are not accepted here. RoundCoordinator
        handles client failures before invoking Strategy.aggregate_fit()
        and only successful FederatedFitResult objects reach this
        aggregation boundary.
        """

        # ----------------------------------------------------
        # Common aggregation validation
        # ----------------------------------------------------

        self._validate_results(results)

        # ----------------------------------------------------
        # FedAvg-specific validation
        # ----------------------------------------------------

        self._validate_sample_counts(results)

        parameter_sets = self._collect_parameter_sets(
            results,
        )

        self._validate_parameter_structure(
            parameter_sets,
        )

        # ----------------------------------------------------
        # Calculate sample weights
        # ----------------------------------------------------

        total_examples = self._calculate_total_examples(
            results,
        )

        weights = self._calculate_weights(
            results,
            total_examples,
        )

        # ----------------------------------------------------
        # Aggregate parameters
        # ----------------------------------------------------

        return self._aggregate_parameter_sets(
            parameter_sets=parameter_sets,
            weights=weights,
        )

    # ========================================================
    # Sample-count validation
    # ========================================================

    @staticmethod
    def _validate_sample_counts(
        results: Mapping[str, FederatedFitResult],
    ) -> None:
        """
        Validate client sample counts used by FedAvg.

        FedAvg requires a strictly positive number of local
        training examples for every contributing client.

        bool is explicitly rejected because Python treats bool as
        a subclass of int.
        """

        for client_id, result in results.items():
            num_examples = result.num_examples

            if isinstance(
                num_examples,
                bool,
            ) or not isinstance(
                num_examples,
                int,
            ):
                raise FederatedLearningError(
                    "FedAvg requires num_examples to be an "
                    "integer for every client. "
                    f"Client '{client_id}' provided "
                    f"{type(num_examples).__name__}."
                )

            if num_examples <= 0:
                raise FederatedLearningError(
                    "FedAvg requires num_examples > 0. "
                    f"Client '{client_id}' provided "
                    f"{num_examples}."
                )

    # ========================================================
    # Parameter extraction
    # ========================================================

    @staticmethod
    def _collect_parameter_sets(
        results: Mapping[str, FederatedFitResult],
    ) -> dict[str, ParameterPayload]:
        """
        Extract defensive parameter references from client results.

        No arrays are copied at this stage because the aggregation
        operation itself creates a new output payload.

        A shallow dictionary is sufficient here because the input
        arrays are treated as read-only throughout aggregation.
        """

        return {
            client_id: result.parameters
            for client_id, result in results.items()
        }

    # ========================================================
    # Parameter structure validation
    # ========================================================

    @staticmethod
    def _validate_parameter_structure(
        parameter_sets: Mapping[str, ParameterPayload],
    ) -> None:
        """
        Validate that all client parameter payloads are structurally
        compatible.

        Validation covers:

        - parameter payload container
        - parameter count
        - NumPy-array type
        - exact shapes
        - exact dtypes
        - finite floating-point values

        FedMed's Phase 3.1 ParameterContract remains the canonical
        model-level parameter contract. This method provides the
        aggregation-level compatibility check needed when combining
        multiple client results.
        """

        if not parameter_sets:
            raise FederatedLearningError(
                "FedAvg cannot validate an empty parameter set."
            )

        reference_client_id = next(
            iter(parameter_sets),
        )

        reference_parameters = parameter_sets[
            reference_client_id
        ]

        if not isinstance(
            reference_parameters,
            (list, tuple),
        ):
            raise FederatedLearningError(
                "Client "
                f"'{reference_client_id}' returned invalid "
                "parameters. Expected a list or tuple of "
                f"NumPy arrays, got "
                f"{type(reference_parameters).__name__}."
            )

        reference_count = len(
            reference_parameters,
        )

        if reference_count == 0:
            raise FederatedLearningError(
                "FedAvg cannot aggregate an empty parameter "
                f"payload from client '{reference_client_id}'."
            )

        # ----------------------------------------------------
        # Validate reference parameter set
        # ----------------------------------------------------

        for index, parameter in enumerate(
            reference_parameters,
        ):
            FedAvgAggregator._validate_parameter_array(
                client_id=reference_client_id,
                index=index,
                parameter=parameter,
            )

        # ----------------------------------------------------
        # Validate every other client against the reference
        # ----------------------------------------------------

        for client_id, parameters in parameter_sets.items():
            if not isinstance(
                parameters,
                (list, tuple),
            ):
                raise FederatedLearningError(
                    "Client "
                    f"'{client_id}' returned invalid parameters. "
                    "Expected a list or tuple of NumPy arrays, "
                    f"got {type(parameters).__name__}."
                )

            if len(parameters) != reference_count:
                raise FederatedLearningError(
                    "FedAvg parameter count mismatch: "
                    f"client '{client_id}' has "
                    f"{len(parameters)} parameters, while "
                    f"client '{reference_client_id}' has "
                    f"{reference_count}."
                )

            for index, parameter in enumerate(
                parameters,
            ):
                FedAvgAggregator._validate_parameter_array(
                    client_id=client_id,
                    index=index,
                    parameter=parameter,
                )

                reference = reference_parameters[
                    index
                ]

                if parameter.shape != reference.shape:
                    raise FederatedLearningError(
                        "FedAvg parameter shape mismatch at "
                        f"index {index}: client "
                        f"'{client_id}' has shape "
                        f"{parameter.shape}, while client "
                        f"'{reference_client_id}' has shape "
                        f"{reference.shape}."
                    )

                if parameter.dtype != reference.dtype:
                    raise FederatedLearningError(
                        "FedAvg parameter dtype mismatch at "
                        f"index {index}: client "
                        f"'{client_id}' has dtype "
                        f"{parameter.dtype}, while client "
                        f"'{reference_client_id}' has dtype "
                        f"{reference.dtype}."
                    )

    @staticmethod
    def _validate_parameter_array(
        client_id: str,
        index: int,
        parameter: object,
    ) -> None:
        """
        Validate one client parameter array.

        Parameter values must remain finite because averaging
        non-finite values would silently contaminate the global
        model.
        """

        if not isinstance(
            parameter,
            np.ndarray,
        ):
            raise FederatedLearningError(
                "FedAvg parameter at index "
                f"{index} for client '{client_id}' "
                "must be a NumPy array, got "
                f"{type(parameter).__name__}."
            )

        # ----------------------------------------------------
        # Floating-point values
        # ----------------------------------------------------

        if np.issubdtype(
            parameter.dtype,
            np.floating,
        ):
            if not np.all(
                np.isfinite(parameter),
            ):
                raise FederatedLearningError(
                    "FedAvg parameter at index "
                    f"{index} for client '{client_id}' "
                    "contains non-finite floating-point "
                    "values."
                )

    # ========================================================
    # Weight calculation
    # ========================================================

    @staticmethod
    def _calculate_total_examples(
        results: Mapping[str, FederatedFitResult],
    ) -> int:
        """
        Calculate the total number of training examples.

        Returns
        -------
        int
            Sum of all successful clients' training samples.
        """

        total_examples = sum(
            result.num_examples
            for result in results.values()
        )

        if total_examples <= 0:
            # This should already be impossible after individual
            # validation, but the invariant is important enough
            # to protect at the point where it is used.
            raise FederatedLearningError(
                "FedAvg total training examples must be "
                f"greater than zero, got {total_examples}."
            )

        return total_examples

    @staticmethod
    def _calculate_weights(
        results: Mapping[str, FederatedFitResult],
        total_examples: int,
    ) -> dict[str, float]:
        """
        Calculate normalized sample-count weights.

        Each weight is:

            client_examples / total_examples

        The weights are validated to ensure they form a valid
        probability-like partition of the total sample mass.
        """

        if total_examples <= 0:
            raise FederatedLearningError(
                "FedAvg cannot calculate weights with a "
                "non-positive total example count."
            )

        weights = {
            client_id: (
                result.num_examples / total_examples
            )
            for client_id, result in results.items()
        }

        weight_sum = sum(
            weights.values(),
        )

        if not np.isclose(
            weight_sum,
            1.0,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise FederatedLearningError(
                "FedAvg client weights must sum to 1.0, "
                f"got {weight_sum}."
            )

        for client_id, weight in weights.items():
            if not np.isfinite(weight):
                raise FederatedLearningError(
                    "FedAvg produced a non-finite weight "
                    f"for client '{client_id}'."
                )

            if weight <= 0.0:
                raise FederatedLearningError(
                    "FedAvg produced a non-positive weight "
                    f"for client '{client_id}': {weight}."
                )

        return weights

    # ========================================================
    # Parameter aggregation
    # ========================================================

    @classmethod
    def _aggregate_parameter_sets(
        cls,
        parameter_sets: Mapping[str, ParameterPayload],
        weights: Mapping[str, float],
    ) -> ParameterPayload:
        """
        Aggregate every model-state entry independently.

        Floating-point arrays:

            weighted average

        Non-floating arrays:

            exact equality required

        A new NumPy array is allocated for every output parameter.
        """

        client_ids = tuple(
            parameter_sets.keys(),
        )

        reference_client_id = client_ids[0]

        reference_parameters = parameter_sets[
            reference_client_id
        ]

        aggregated: ParameterPayload = []

        for index in range(
            len(reference_parameters),
        ):
            arrays = [
                parameter_sets[client_id][index]
                for client_id in client_ids
            ]

            reference = arrays[0]

            if np.issubdtype(
                reference.dtype,
                np.floating,
            ):
                aggregated.append(
                    cls._aggregate_floating_parameter(
                        arrays=arrays,
                        client_ids=client_ids,
                        weights=weights,
                        index=index,
                    )
                )

            else:
                aggregated.append(
                    cls._aggregate_non_floating_parameter(
                        arrays=arrays,
                        client_ids=client_ids,
                        index=index,
                    )
                )

        return aggregated

    @staticmethod
    def _aggregate_floating_parameter(
        arrays: list[np.ndarray],
        client_ids: tuple[str, ...],
        weights: Mapping[str, float],
        index: int,
    ) -> np.ndarray:
        """
        Compute a weighted average for one floating-point parameter.

        Accumulation uses float64 when the source dtype is a
        floating-point type.

        The final result is converted back to the exact reference
        dtype so it remains compatible with the existing
        ParameterContract.

        Input arrays are never modified.
        """

        reference = arrays[0]

        # Use float64 for accumulation to reduce numerical error
        # when multiple client models are combined. The final
        # parameter is restored to the reference dtype.
        accumulator = np.zeros(
            reference.shape,
            dtype=np.float64,
        )

        for client_id, parameter in zip(
            client_ids,
            arrays,
        ):
            weight = weights[client_id]

            # The multiplication creates a temporary array.
            # The client-owned parameter itself is never modified.
            accumulator += (
                parameter.astype(
                    np.float64,
                    copy=False,
                )
                * weight
            )

        if not np.all(
            np.isfinite(accumulator),
        ):
            raise FederatedLearningError(
                "FedAvg produced non-finite values while "
                f"aggregating floating-point parameter "
                f"at index {index}."
            )

        try:
            result = accumulator.astype(
                reference.dtype,
                copy=True,
            )
        except (TypeError, ValueError) as exc:
            raise FederatedLearningError(
                "FedAvg could not restore aggregated "
                f"parameter at index {index} to dtype "
                f"{reference.dtype}."
            ) from exc

        # The cast can theoretically overflow for unusually
        # constrained floating-point types. Validate the final
        # representation as well.
        if not np.all(
            np.isfinite(result),
        ):
            raise FederatedLearningError(
                "FedAvg aggregated parameter at index "
                f"{index} contains non-finite values after "
                f"conversion to dtype {reference.dtype}."
            )

        return result

    @staticmethod
    def _aggregate_non_floating_parameter(
        arrays: list[np.ndarray],
        client_ids: tuple[str, ...],
        index: int,
    ) -> np.ndarray:
        """
        Aggregate one non-floating model-state entry.

        Non-floating state is not averaged.

        Every participating client must provide exactly the same
        value. The resulting array is a defensive copy of the
        reference value.

        This avoids mathematically meaningless operations such as
        averaging integer counters or boolean state.
        """

        reference = arrays[0]

        for client_id, parameter in zip(
            client_ids[1:],
            arrays[1:],
        ):
            if not np.array_equal(
                parameter,
                reference,
            ):
                raise FederatedLearningError(
                    "FedAvg cannot aggregate conflicting "
                    "non-floating parameter state at "
                    f"index {index}: client "
                    f"'{client_id}' differs from reference "
                    f"client '{client_ids[0]}'."
                )

        return reference.copy()