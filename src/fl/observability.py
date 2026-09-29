"""
Structured federated round observability for FedMed.

Phase 3.5
---------

This module provides a small, framework-independent representation of
what happened during one completed federated round.

Why this module exists
----------------------

Before this module, round-level information was reachable only through
three different shapes:

    RoundResult
        Structured, but carries live client result objects
        (including full client parameter payloads) and per-client
        metric mappings.

    return values / logs
        Scattered across ``RoundCoordinator`` and the runtime
        adapters, and not comparable across rounds.

    Flower runtime output
        ``MetricRecord`` values printed by the Flower adapter.

None of those is a compact, serializable record of a round that can be
inspected, compared, or later persisted. This module adds that record
without changing any existing abstraction.

Responsibilities
----------------

This module is responsible for:

- describing one completed federated round as a flat, immutable record
- separating successful clients from failed clients
- recording sample counts and the round-level training loss
- recording round-level evaluation loss and accuracy when available
- recording an optional deterministic parameter fingerprint
- deriving a summary from an existing ``RoundExecution``

This module intentionally does NOT:

- execute federated rounds
- select clients
- implement or modify aggregation
- implement or modify federation policy
- decide whether a client failure aborts a round
- change the failure policy
- change checkpoint contents
- implement networking or transport
- import Flower or any runtime/transport API
- introduce a logging framework or external dependency

Architecture
------------

    RoundExecution
           |
           |  result  (RoundResult)
           |  aggregated_parameters (ParameterPayload)
           v
    FederatedRoundSummary
           |
           +-- round number and status
           +-- participant accounting
           +-- training / evaluation observations
           +-- parameter fingerprint

The summary is *derived* from ``RoundExecution`` rather than stored
alongside it. ``RoundExecution`` remains the operational output of a
round, and the summary is a read-only projection of it. Nothing in the
round lifecycle needs to know the summary exists.

Relationship to existing abstractions
--------------------------------------

``RoundResult`` is deliberately not replaced or extended. The summary
reuses the existing vocabulary:

    selected_clients
        ``RoundResult.selected_clients``

    successful_clients
        ``RoundResult.successful_clients``

    failed_clients
        ``RoundResult.failed_clients`` (reuses ``ClientFailure``)

    evaluation_failures
        ``RoundResult.evaluation_failures``

    training_examples
        summed from ``RoundResult.fit_results``

    evaluation_examples
        summed from ``RoundResult.evaluation_results``

The summary adds only what ``RoundResult`` does not express: a single
round-level training loss, a single round-level evaluation observation,
and a parameter fingerprint.

Aggregation and policy
----------------------

The summary is a *reporting* projection. It never influences
aggregation and never influences which clients participate.

Sample counts are simple sums over already-recorded client results.

The round-level training loss is a sample-weighted mean over the
successful clients' ``FederatedFitResult.final_loss``. The aggregated
parameters were already produced by the Strategy/Aggregator before the
summary exists, so this reduction cannot influence them.

Round-level evaluation loss and accuracy follow the same sample-weighted
mean over the recorded per-client evaluation results. This is a
reduction for observation only; it mirrors the weighting the FedAvg
strategy already uses for evaluation, and it never influences which
clients are selected, which parameters are aggregated, or whether a
round succeeds.

When the caller already holds a round-level evaluation aggregate, that
aggregate can be supplied to ``from_round_execution(evaluation=...)``
and is then used verbatim, so a runtime that already aggregated
evaluation through its Strategy reports exactly the same numbers.

Fingerprinting
--------------

``parameter_fingerprint()`` computes a deterministic, truncated
SHA-256 digest over a parameter payload. Identical payloads always
produce identical fingerprints, which makes the fingerprint usable as a
cheap "did the model actually change?" signal.

Checkpoint compatibility
------------------------

A summary is derived on demand from round history. It is never part of
``FederationCheckpoint`` and is never restored. A server restored from
a checkpoint resumes with an empty round history, and therefore with
an empty summary history, exactly as it already does for
``RoundExecution``.
"""

from __future__ import annotations

import hashlib
import math

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import numpy as np

from src.common.exceptions import FederatedLearningError
from src.fl.client import FederatedEvaluateResult, FederatedFitResult
from src.fl.rounds import (
    ClientFailure,
    RoundExecution,
    RoundResult,
    RoundState,
)
from src.fl.strategy import FedAvgEvaluationResult


# ============================================================
# Module constants
# ============================================================


#: Result key produced by ``src.training.metrics.Accuracy``.
#:
#: The summary reads accuracy from an already-aggregated evaluation
#: metric mapping using this key. A round whose evaluation metrics do
#: not include accuracy simply reports ``evaluation_accuracy=None``
#: rather than guessing.
ACCURACY_METRIC_KEY: Final[str] = "accuracy"


#: Number of hex characters retained from the SHA-256 digest.
#:
#: Sixteen hex characters (64 bits) are enough to detect parameter
#: changes while keeping the record small.
FINGERPRINT_LENGTH: Final[int] = 16


# ============================================================
# Parameter fingerprint
# ============================================================


def parameter_fingerprint(
    parameters: Sequence[np.ndarray],
) -> str:
    """
    Compute a deterministic fingerprint for a parameter payload.

    The digest is computed over, for every parameter in order, its
    shape, its dtype, and its contiguous raw bytes. Ordering is part of
    the contract, so two payloads with the same values in a different
    order produce different fingerprints.

    The function is pure and framework-independent. It performs no
    aggregation, no model inspection, and no I/O.

    Parameters
    ----------
    parameters:
        A FedMed ``ParameterPayload``.

    Returns
    -------
    str
        A deterministic hexadecimal digest prefix.

    Raises
    ------
    FederatedLearningError
        If the payload is not a sequence of NumPy arrays.
    """

    if isinstance(parameters, (str, bytes)):
        raise FederatedLearningError(
            "parameter_fingerprint() requires a sequence of NumPy "
            "arrays, not a string."
        )

    if not isinstance(parameters, Sequence):
        raise FederatedLearningError(
            "parameter_fingerprint() requires a sequence of NumPy "
            f"arrays, got {type(parameters).__name__}."
        )

    digest = hashlib.sha256()

    for parameter in parameters:

        array = np.asarray(parameter)

        digest.update(str(array.shape).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())

    return digest.hexdigest()[:FINGERPRINT_LENGTH]


# ============================================================
# Round summary
# ============================================================


@dataclass(frozen=True)
class FederatedRoundSummary:
    """
    Immutable, flat record of one completed federated round.

    The summary answers: who was selected, who succeeded, who failed,
    how much data was used, and what the round-level observations were.

    Attributes
    ----------
    round_number:
        One-based federated round number.

    status:
        Final state of the round. Derived from ``RoundResult.status``.

    selected_clients:
        Client IDs selected for the round's training operation.

    successful_clients:
        Client IDs whose local training completed successfully.

    failed_clients:
        Structured *training* failures. Reuses the existing
        ``ClientFailure`` type from ``src.fl.rounds``.

    evaluation_failures:
        Structured *evaluation* failures. Kept separate from
        ``failed_clients`` for the same reason ``RoundResult`` keeps
        them separate.

    training_examples:
        Total local examples processed by the successful clients.

    training_loss:
        Sample-weighted mean of the successful clients' final training
        loss, or ``None`` when no client trained successfully.

    evaluation_examples:
        Total local examples evaluated. Zero when the round did not
        evaluate.

    evaluation_loss:
        Round-level evaluation loss, or ``None`` when no aggregated
        evaluation observation is available.

    evaluation_accuracy:
        Round-level evaluation accuracy, or ``None`` when no aggregated
        accuracy value is available.

    parameter_fingerprint:
        Optional deterministic fingerprint of the round's aggregated
        parameters.

    Invariants
    ----------
    1. ``round_number`` is an integer ``>= 1``.

    2. Every client ID is a non-empty string and no client ID appears
       twice within the same client set.

    3. ``successful_clients`` is a subset of ``selected_clients``.

    4. Training failure IDs are a subset of ``selected_clients`` and
       are disjoint from ``successful_clients``. A client that failed
       is never reported as successful.

    5. ``failed_clients`` contains only ``TRAINING`` failures and
       ``evaluation_failures`` contains only ``EVALUATING`` failures.

    6. ``training_examples`` is ``>= 1`` exactly when at least one
       client trained successfully, and ``0`` otherwise.

    7. ``evaluation_examples`` is ``>= 0``; it is ``0`` exactly when
       ``evaluation_loss`` and ``evaluation_accuracy`` are both
       ``None``.

    8. Every recorded metric is a finite, non-negative float.
       ``evaluation_accuracy``, when present, is within ``[0.0, 1.0]``.

    Notes
    -----
    The summary stores no Flower ``Message``, no client parameter
    payload, and no heavyweight client result object. Every field is a
    scalar or an immutable container of small values, which keeps the
    record cheap to retain and simple to persist later.
    """

    round_number: int
    status: RoundState

    selected_clients: tuple[str, ...]
    successful_clients: tuple[str, ...]

    failed_clients: tuple[ClientFailure, ...] = ()
    evaluation_failures: tuple[ClientFailure, ...] = ()

    training_examples: int = 0
    training_loss: float | None = None

    evaluation_examples: int = 0
    evaluation_loss: float | None = None
    evaluation_accuracy: float | None = None

    parameter_fingerprint: str | None = None

    def __post_init__(self) -> None:
        """Validate and freeze the recorded round observations."""

        # ----------------------------------------------------
        # Round identity
        # ----------------------------------------------------

        self._validate_round_number(
            self.round_number,
        )

        if not isinstance(self.status, RoundState):
            raise FederatedLearningError(
                "FederatedRoundSummary.status must be a RoundState, "
                f"got {type(self.status).__name__}."
            )

        # ----------------------------------------------------
        # Normalize immutable sequence fields
        # ----------------------------------------------------

        selected = tuple(self.selected_clients)
        successful = tuple(self.successful_clients)
        failures = tuple(self.failed_clients)
        evaluation_failures = tuple(
            self.evaluation_failures
        )

        # ----------------------------------------------------
        # Client ID validation
        # ----------------------------------------------------

        self._validate_client_ids(
            selected,
            "selected_clients",
        )

        self._validate_client_ids(
            successful,
            "successful_clients",
        )

        selected_set = set(selected)
        successful_set = set(successful)

        if not successful_set.issubset(selected_set):
            raise FederatedLearningError(
                "Every successful client must be present in "
                "selected_clients."
            )

        # ----------------------------------------------------
        # Training failures
        # ----------------------------------------------------

        training_failure_ids = (
            self._validate_failures(
                failures,
                "failed_clients",
                RoundState.TRAINING,
            )
        )

        if not training_failure_ids.issubset(
            selected_set
        ):
            raise FederatedLearningError(
                "Every failed client must be present in "
                "selected_clients."
            )

        if successful_set.intersection(
            training_failure_ids
        ):
            raise FederatedLearningError(
                "A failed client cannot be reported as a "
                "successful client."
            )

        # ----------------------------------------------------
        # Evaluation failures
        # ----------------------------------------------------

        # An evaluation failure may belong to a client that trained
        # successfully, so this set is intentionally not compared
        # against successful_clients. It must still be a subset of
        # the round's selected clients.
        self._validate_failures(
            evaluation_failures,
            "evaluation_failures",
            RoundState.EVALUATING,
        )

        evaluation_failure_ids = {
            failure.client_id
            for failure in evaluation_failures
        }

        if not evaluation_failure_ids.issubset(
            selected_set
        ):
            raise FederatedLearningError(
                "Every evaluation failure client must be present "
                "in selected_clients."
            )

        # ----------------------------------------------------
        # Training observations
        # ----------------------------------------------------

        training_examples = self._validate_count(
            self.training_examples,
            "training_examples",
        )

        if successful_set and training_examples < 1:
            raise FederatedLearningError(
                "training_examples must be >= 1 when at least one "
                "client trained successfully."
            )

        if not successful_set and training_examples != 0:
            raise FederatedLearningError(
                "training_examples must be 0 when no client trained "
                "successfully."
            )

        training_loss = self._validate_optional_metric(
            self.training_loss,
            "training_loss",
        )

        if training_loss is not None and not successful_set:
            raise FederatedLearningError(
                "training_loss cannot be recorded when no client "
                "trained successfully."
            )

        if successful_set and training_loss is None:
            raise FederatedLearningError(
                "training_loss must be recorded when at least one "
                "client trained successfully."
            )

        # ----------------------------------------------------
        # Evaluation observations
        # ----------------------------------------------------

        evaluation_examples = self._validate_count(
            self.evaluation_examples,
            "evaluation_examples",
        )

        evaluation_loss = self._validate_optional_metric(
            self.evaluation_loss,
            "evaluation_loss",
        )

        evaluation_accuracy = self._validate_optional_metric(
            self.evaluation_accuracy,
            "evaluation_accuracy",
        )

        if (
            evaluation_accuracy is not None
            and not 0.0 <= evaluation_accuracy <= 1.0
        ):
            raise FederatedLearningError(
                "evaluation_accuracy must be within [0.0, 1.0], got "
                f"{evaluation_accuracy}."
            )

        if evaluation_examples == 0:
            if (
                evaluation_loss is not None
                or evaluation_accuracy is not None
            ):
                raise FederatedLearningError(
                    "Evaluation metrics cannot be recorded when "
                    "evaluation_examples is 0."
                )

        elif evaluation_loss is None and evaluation_accuracy is None:
            raise FederatedLearningError(
                "At least one evaluation metric must be recorded "
                "when evaluation_examples > 0."
            )

        # ----------------------------------------------------
        # Parameter fingerprint
        # ----------------------------------------------------

        fingerprint = self.parameter_fingerprint

        if fingerprint is not None:
            if not isinstance(fingerprint, str):
                raise FederatedLearningError(
                    "parameter_fingerprint must be a string or None, "
                    f"got {type(fingerprint).__name__}."
                )

            if not fingerprint.strip():
                raise FederatedLearningError(
                    "parameter_fingerprint cannot be empty."
                )

        # ----------------------------------------------------
        # Freeze normalized data
        # ----------------------------------------------------

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
            "evaluation_failures",
            evaluation_failures,
        )

    # ========================================================
    # Derived views
    # ========================================================

    @property
    def failed_client_ids(self) -> tuple[str, ...]:
        """
        Return the IDs of clients whose training failed.

        This is a convenience view over ``failed_clients``. Evaluation
        failures are reported separately by
        ``evaluation_failed_client_ids``.
        """

        return tuple(
            failure.client_id
            for failure in self.failed_clients
        )

    @property
    def evaluation_failed_client_ids(self) -> tuple[str, ...]:
        """Return the IDs of clients whose evaluation failed."""

        return tuple(
            failure.client_id
            for failure in self.evaluation_failures
        )

    @property
    def participation_count(self) -> int:
        """Return the number of clients that trained successfully."""

        return len(self.successful_clients)

    @property
    def dropout_count(self) -> int:
        """Return the number of clients that failed to train."""

        return len(self.failed_clients)

    @property
    def evaluated(self) -> bool:
        """Return whether this round recorded evaluation."""

        return self.evaluation_examples > 0

    # ========================================================
    # Construction
    # ========================================================

    @classmethod
    def from_round_execution(
        cls,
        execution: RoundExecution,
        *,
        evaluation: FedAvgEvaluationResult | None = None,
        include_fingerprint: bool = True,
    ) -> "FederatedRoundSummary":
        """
        Derive a summary from a completed ``RoundExecution``.

        Parameters
        ----------
        execution:
            The operational output of one completed round.

        evaluation:
            Optional round-level evaluation produced by the Strategy's
            existing ``aggregate_evaluate`` boundary.

            When supplied, its loss and metrics are recorded verbatim.
            When omitted, the round-level evaluation loss and accuracy
            are derived as a sample-weighted reduction over
            ``RoundResult.evaluation_results``.

            Both paths are reporting-only. Supplying the Strategy's own
            aggregate keeps evaluation reporting identical to the
            runtime's evaluation reporting.

        include_fingerprint:
            Whether to compute the aggregated-parameter fingerprint.

        Returns
        -------
        FederatedRoundSummary
            An immutable, validated round summary.

        Raises
        ------
        FederatedLearningError
            If the execution is invalid, or the supplied evaluation
            aggregate is inconsistent with the round's client set.

        Notes
        -----
        The derived summary is a read-only projection. Building it does
        not modify the execution or the client results it reads.
        """

        if not isinstance(execution, RoundExecution):
            raise FederatedLearningError(
                "from_round_execution() requires a RoundExecution, "
                f"got {type(execution).__name__}."
            )

        result = execution.result

        training_examples, training_loss = (
            cls._training_observations(result)
        )

        evaluation_examples = cls._evaluation_examples(result)

        (
            evaluation_loss,
            evaluation_accuracy,
        ) = cls._evaluation_observations(
            result,
            evaluation=evaluation,
        )

        fingerprint = (
            parameter_fingerprint(
                execution.aggregated_parameters,
            )
            if include_fingerprint
            else None
        )

        return cls(
            round_number=result.round_number,
            status=result.status,
            selected_clients=result.selected_clients,
            successful_clients=result.successful_clients,
            failed_clients=result.failed_clients,
            evaluation_failures=(
                result.evaluation_failures
            ),
            training_examples=training_examples,
            training_loss=training_loss,
            evaluation_examples=evaluation_examples,
            evaluation_loss=evaluation_loss,
            evaluation_accuracy=evaluation_accuracy,
            parameter_fingerprint=fingerprint,
        )

    # ========================================================
    # Observation helpers
    # ========================================================

    @staticmethod
    def _training_observations(
        result: RoundResult,
    ) -> tuple[int, float | None]:
        """
        Return total training examples and the sample-weighted loss.

        The loss reduction is used for reporting only. It runs after
        aggregation has already produced the round's parameters and
        cannot influence them.
        """

        fit_results: Mapping[
            str,
            FederatedFitResult,
        ] = result.fit_results

        if not fit_results:
            return 0, None

        total_examples = 0
        weighted_loss = 0.0

        for fit_result in fit_results.values():

            num_examples = (
                fit_result.num_examples
            )

            if num_examples <= 0:
                raise FederatedLearningError(
                    "Cannot summarize a round whose successful "
                    "clients reported no training examples."
                )

            loss = float(fit_result.final_loss)

            if not math.isfinite(loss) or loss < 0.0:
                raise FederatedLearningError(
                    "A client reported a non-finite or negative "
                    f"final training loss: {loss}."
                )

            total_examples += num_examples
            weighted_loss += loss * num_examples

        if total_examples <= 0:
            return 0, None

        return total_examples, weighted_loss / total_examples

    @staticmethod
    def _evaluation_examples(
        result: RoundResult,
    ) -> int:
        """Return the total number of locally evaluated examples."""

        evaluation_results: Mapping[
            str,
            FederatedEvaluateResult,
        ] = result.evaluation_results

        if not evaluation_results:
            return 0

        total = 0

        for evaluate_result in evaluation_results.values():
            total += int(evaluate_result.num_examples)

        return total

    @staticmethod
    def _evaluation_observations(
        result: RoundResult,
        *,
        evaluation: FedAvgEvaluationResult | None,
    ) -> tuple[float | None, float | None]:
        """
        Return round-level evaluation loss and accuracy.

        When the caller supplies the Strategy's aggregate, that
        aggregate is used verbatim. Otherwise the summary falls back to
        a sample-weighted reduction over the recorded per-client
        evaluation results.
        """

        evaluation_results: Mapping[
            str,
            FederatedEvaluateResult,
        ] = result.evaluation_results

        if not evaluation_results:
            if evaluation is not None:
                raise FederatedLearningError(
                    "An aggregated evaluation result was supplied "
                    "for a round that performed no evaluation."
                )

            return None, None

        if evaluation is not None:

            if not isinstance(
                evaluation,
                FedAvgEvaluationResult,
            ):
                raise FederatedLearningError(
                    "evaluation must be a FedAvgEvaluationResult "
                    "or None, got "
                    f"{type(evaluation).__name__}."
                )

            accuracy = evaluation.metrics.get(
                ACCURACY_METRIC_KEY,
            )

            return float(evaluation.loss), (
                None
                if accuracy is None
                else float(accuracy)
            )

        # ----------------------------------------------------
        # Fallback: sample-weighted reduction over recorded results
        # ----------------------------------------------------

        total_examples = 0
        weighted_loss = 0.0
        accumulated: dict[str, float] = {}
        weights: dict[str, int] = {}

        for evaluate_result in evaluation_results.values():

            num_examples = int(
                evaluate_result.num_examples
            )

            if num_examples <= 0:
                raise FederatedLearningError(
                    "Cannot summarize a round whose successful "
                    "clients reported no evaluation examples."
                )

            loss = float(evaluate_result.loss)

            if not math.isfinite(loss) or loss < 0.0:
                raise FederatedLearningError(
                    "A client reported a non-finite or negative "
                    f"evaluation loss: {loss}."
                )

            total_examples += num_examples
            weighted_loss += loss * num_examples

            for key, value in evaluate_result.metrics.items():

                numeric = float(value)

                if not math.isfinite(numeric):
                    raise FederatedLearningError(
                        "A client reported a non-finite "
                        f"evaluation metric '{key}': {numeric}."
                    )

                accumulated[key] = (
                    accumulated.get(key, 0.0)
                    + numeric * num_examples
                )

                weights[key] = (
                    weights.get(key, 0) + num_examples
                )

        if total_examples <= 0:
            return None, None

        metrics = {
            key: accumulated[key] / weights[key]
            for key in accumulated
        }

        return (
            weighted_loss / total_examples,
            metrics.get(ACCURACY_METRIC_KEY),
        )

    # ========================================================
    # Validation helpers
    # ========================================================

    @staticmethod
    def _validate_round_number(
        round_number: int,
    ) -> None:
        """Validate the one-based round number."""

        if isinstance(round_number, bool) or not isinstance(
            round_number, int
        ):
            raise FederatedLearningError(
                "FederatedRoundSummary.round_number must be an "
                f"integer, got {type(round_number).__name__}."
            )

        if round_number < 1:
            raise FederatedLearningError(
                "FederatedRoundSummary.round_number must be >= 1, "
                f"got {round_number}."
            )

    @staticmethod
    def _validate_client_ids(
        client_ids: tuple[str, ...],
        field_name: str,
    ) -> None:
        """Validate one tuple of client identifiers."""

        for client_id in client_ids:

            if not isinstance(client_id, str):
                raise FederatedLearningError(
                    f"{field_name} must contain only strings, got "
                    f"{type(client_id).__name__}."
                )

            if not client_id.strip():
                raise FederatedLearningError(
                    f"{field_name} cannot contain empty client IDs."
                )

        if len(set(client_ids)) != len(client_ids):
            raise FederatedLearningError(
                f"{field_name} contains duplicate client IDs."
            )

    @staticmethod
    def _validate_failures(
        failures: tuple[ClientFailure, ...],
        field_name: str,
        expected_phase: RoundState,
    ) -> set[str]:
        """
        Validate one tuple of structured client failures.

        Returns the set of failed client IDs.
        """

        client_ids: list[str] = []

        for failure in failures:

            if not isinstance(
                failure,
                ClientFailure,
            ):
                raise FederatedLearningError(
                    f"{field_name} must contain only "
                    "ClientFailure objects."
                )

            if failure.phase is not expected_phase:
                raise FederatedLearningError(
                    f"{field_name} may contain only "
                    f"{expected_phase.value} failures."
                )

            client_ids.append(failure.client_id)

        if len(set(client_ids)) != len(client_ids):
            raise FederatedLearningError(
                f"{field_name} contains duplicate client IDs."
            )

        return set(client_ids)

    @staticmethod
    def _validate_count(
        value: int,
        field_name: str,
    ) -> int:
        """Validate a non-negative example count."""

        if isinstance(value, bool) or not isinstance(
            value, int
        ):
            raise FederatedLearningError(
                f"{field_name} must be an integer, got "
                f"{type(value).__name__}."
            )

        if value < 0:
            raise FederatedLearningError(
                f"{field_name} must be >= 0, got {value}."
            )

        return value

    @staticmethod
    def _validate_optional_metric(
        value: float | None,
        field_name: str,
    ) -> float | None:
        """Validate an optional finite, non-negative metric."""

        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise FederatedLearningError(
                f"{field_name} must be a number or None, got "
                f"{type(value).__name__}."
            )

        numeric = float(value)

        if not math.isfinite(numeric):
            raise FederatedLearningError(
                f"{field_name} must be finite, got {value}."
            )

        if numeric < 0.0:
            raise FederatedLearningError(
                f"{field_name} must be >= 0, got {value}."
            )

        return numeric


__all__ = [
    "ACCURACY_METRIC_KEY",
    "FINGERPRINT_LENGTH",
    "FederatedRoundSummary",
    "parameter_fingerprint",
]
