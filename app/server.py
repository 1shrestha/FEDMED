"""
Flower ServerApp runtime adapter for FedMed.

Flower-specific transport/runtime behavior lives here. The
framework-independent FedMed core remains under ``src/`` and is not
modified by this adapter.

Boundary:

    Flower ServerApp / Grid
            |
            v
    FedMedFlowerStrategy
            |
            +--> FedMed FederatedStrategy
            |        |
            |        +--> Aggregator
            |               |
            |               +--> FedAvgAggregator
            |
            v
    Flower NumPyClient compatibility boundary

Important
---------
FedMed uses Flower's ``NumPyClient.to_client()`` compatibility adapter
on the client side.

Flower 1.34.0's compatibility layer converts incoming TRAIN/EVALUATE
messages using the legacy RecordDict keys:

    fitins.parameters
    fitins.config

    evaluateins.parameters
    evaluateins.config

Therefore this module intentionally uses those keys when constructing
messages for the current NumPyClient-based FedMed adapter.

These keys are Flower transport details only. They are not FedMed
domain concepts.

FedMed's framework-independent Strategy/Aggregator contracts remain
unchanged:

    Mapping[str, FederatedFitResult]
                    |
                    v
            FedAvgStrategy
                    |
                    v
             Aggregator
                    |
                    v
             ParameterPayload
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import Strategy

from src.aggregation.fedavg import FedAvgAggregator
from src.common.exceptions import FederatedLearningError
from src.fl.client import (
    FederatedEvaluateResult,
    FederatedFitResult,
)
from src.fl.parameters import ParameterPayload
from src.fl.strategy import (
    FedAvgEvaluationResult,
    FedAvgStrategy,
    FederatedStrategy,
)


# ======================================================================
# Flower NumPyClient compatibility RecordDict keys
# ======================================================================
#
# app/client.py uses:
#
#     FedMedNumPyClient(...).to_client()
#
# Flower's compatibility layer therefore expects:
#
# Training request:
#
#     fitins.parameters
#     fitins.config
#
# Training response:
#
#     fitres.parameters
#     fitres.num_examples
#     fitres.metrics
#     fitres.status
#
# Evaluation request:
#
#     evaluateins.parameters
#     evaluateins.config
#
# Evaluation response:
#
#     evaluateres.loss
#     evaluateres.num_examples
#     evaluateres.metrics
#     evaluateres.status
#
# These are Flower compatibility-boundary keys only.
# ======================================================================


FIT_PARAMETERS_KEY = "fitins.parameters"
FIT_CONFIG_KEY = "fitins.config"

FITRES_PARAMETERS_KEY = "fitres.parameters"
FITRES_NUM_EXAMPLES_KEY = "fitres.num_examples"
FITRES_METRICS_KEY = "fitres.metrics"
FITRES_STATUS_KEY = "fitres.status"

EVALUATE_PARAMETERS_KEY = "evaluateins.parameters"
EVALUATE_CONFIG_KEY = "evaluateins.config"

EVALUATERES_LOSS_KEY = "evaluateres.loss"
EVALUATERES_NUM_EXAMPLES_KEY = "evaluateres.num_examples"
EVALUATERES_METRICS_KEY = "evaluateres.metrics"
EVALUATERES_STATUS_KEY = "evaluateres.status"


NUM_EXAMPLES_KEY = "num_examples"


InitialParametersFactory = Callable[[Any], ParameterPayload]
StrategyFactory = Callable[[Any], FedAvgStrategy]


# ======================================================================
# Flower strategy adapter
# ======================================================================


class FedMedFlowerStrategy(Strategy):
    """
    Flower ServerApp strategy adapter backed by FedMed.

    Flower owns:
        - ServerApp execution
        - Grid communication
        - Message transport
        - ArrayRecord/ConfigRecord transport
        - NumPyClient compatibility conversion
        - node lifecycle

    FedMed owns:
        - FederatedClient
        - FedAvgStrategy
        - Aggregator
        - framework-independent parameter/result contracts

    The adapter performs only the conversion required at the
    Flower/FedMed boundary.
    """

    def __init__(
        self,
        fedmed_strategy: FedAvgStrategy,
        *,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_available_nodes: int = 1,
    ) -> None:
        if not isinstance(
            fedmed_strategy,
            FedAvgStrategy,
        ):
            raise FederatedLearningError(
                "FedMedFlowerStrategy requires a FedAvgStrategy, "
                f"got {type(fedmed_strategy).__name__}."
            )

        self._fraction_train = self._validate_fraction(
            fraction_train,
            "fraction_train",
        )

        self._fraction_evaluate = self._validate_fraction(
            fraction_evaluate,
            "fraction_evaluate",
        )

        if (
            not isinstance(min_available_nodes, int)
            or isinstance(min_available_nodes, bool)
            or min_available_nodes < 1
        ):
            raise FederatedLearningError(
                "min_available_nodes must be a positive integer."
            )

        self._min_available_nodes = min_available_nodes
        self._fedmed_strategy = fedmed_strategy

        print(
            "[FedMed] Flower strategy adapter created: "
            f"{type(self).__name__}"
        )

    # ==================================================================
    # Public properties
    # ==================================================================

    @property
    def fedmed_strategy(self) -> FedAvgStrategy:
        """Return the framework-independent FedMed Strategy."""
        return self._fedmed_strategy

    # ==================================================================
    # Training configuration
    # ==================================================================

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """
        Construct TRAIN messages for Flower's NumPyClient adapter.

        Important:
            ``NumPyClient.to_client()`` requires the legacy
            ``fitins.parameters`` and ``fitins.config`` keys.
        """

        print(
            "[FedMed DEBUG] configure_train ENTER",
            flush=True,
        )

        selected_nodes = self._select_nodes(
            grid,
            self._fraction_train,
        )

        print(
            "[FedMed DEBUG] selected training nodes:",
            selected_nodes,
            flush=True,
        )

        train_config = ConfigRecord(dict(config))
        train_config["server-round"] = server_round

        record = RecordDict(
            {
                FIT_PARAMETERS_KEY: arrays,
                FIT_CONFIG_KEY: train_config,
            }
        )

        messages = self._construct_messages(
            record,
            selected_nodes,
            MessageType.TRAIN,
        )

        print(
            "[FedMed DEBUG] constructed "
            f"{len(messages)} TRAIN message(s)",
            flush=True,
        )

        return messages

    # ==================================================================
    # Evaluation configuration
    # ==================================================================

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """
        Construct EVALUATE messages for Flower's NumPyClient adapter.

        Important:
            ``NumPyClient.to_client()`` requires the legacy
            ``evaluateins.parameters`` and ``evaluateins.config`` keys.
        """

        selected_nodes = self._select_nodes(
            grid,
            self._fraction_evaluate,
        )

        print(
            "[FedMed DEBUG] selected evaluation nodes:",
            selected_nodes,
            flush=True,
        )

        evaluate_config = ConfigRecord(dict(config))
        evaluate_config["server-round"] = server_round

        record = RecordDict(
            {
                EVALUATE_PARAMETERS_KEY: arrays,
                EVALUATE_CONFIG_KEY: evaluate_config,
            }
        )

        messages = self._construct_messages(
            record,
            selected_nodes,
            MessageType.EVALUATE,
        )

        print(
            "[FedMed DEBUG] constructed "
            f"{len(messages)} EVALUATE message(s)",
            flush=True,
        )

        return messages

    # ==================================================================
    # Training aggregation
    # ==================================================================

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """
        Convert Flower training replies to FedMed results and delegate
        parameter aggregation to the FedMed Strategy/Aggregator.
        """

        replies = list(replies)

        if not replies:
            return None, None

        results: dict[str, FederatedFitResult] = {}
        seen_nodes: set[int] = set()

        for reply in replies:
            if reply.has_error():
                print(
                    "[FedMed DEBUG] ignoring failed training reply from "
                    f"node {reply.metadata.src_node_id}: "
                    f"{reply.error.reason if reply.error else 'unknown error'}",
                    flush=True,
                )
                continue

            node_id = int(reply.metadata.src_node_id)

            if node_id in seen_nodes:
                raise FederatedLearningError(
                    f"Duplicate training reply from node {node_id}."
                )

            seen_nodes.add(node_id)

            result = self._fit_result_from_message(
                reply,
            )

            results[str(node_id)] = result

        if not results:
            return None, None

        print(
            "[FedMed] Delegating Flower training aggregation to "
            f"FedMed Strategy for round {server_round}.",
            flush=True,
        )

        aggregated_parameters = (
            self._fedmed_strategy.aggregate_fit(
                results,
                round_number=server_round,
            )
        )

        arrays = ArrayRecord.from_numpy_ndarrays(
            self._copy_parameters(
                aggregated_parameters,
            )
        )

        observations = [
            (
                result.num_examples,
                result.metrics,
            )
            for result in results.values()
        ]

        metrics = MetricRecord(
            self._weighted_metrics(
                observations,
            )
        )

        return arrays, metrics

    # ==================================================================
    # Evaluation aggregation
    # ==================================================================

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        """
        Convert Flower evaluation replies to FedMed results and
        delegate evaluation aggregation to the FedMed Strategy.
        """

        replies = list(replies)

        if not replies:
            return None

        results: dict[str, FederatedEvaluateResult] = {}
        seen_nodes: set[int] = set()

        for reply in replies:
            if reply.has_error():
                print(
                    "[FedMed DEBUG] ignoring failed evaluation reply from "
                    f"node {reply.metadata.src_node_id}: "
                    f"{reply.error.reason if reply.error else 'unknown error'}",
                    flush=True,
                )
                continue

            node_id = int(reply.metadata.src_node_id)

            if node_id in seen_nodes:
                raise FederatedLearningError(
                    f"Duplicate evaluation reply from node {node_id}."
                )

            seen_nodes.add(node_id)

            result = self._evaluate_result_from_message(
                reply,
            )

            results[str(node_id)] = result

        if not results:
            return None

        print(
            "[FedMed] Delegating Flower evaluation aggregation to "
            f"FedMed Strategy for round {server_round}.",
            flush=True,
        )

        aggregated: FedAvgEvaluationResult = (
            self._fedmed_strategy.aggregate_evaluate(
                results,
                round_number=server_round,
            )
        )

        total_examples = sum(
            result.num_examples
            for result in results.values()
        )

        metric_payload: dict[str, float] = {
            NUM_EXAMPLES_KEY: float(total_examples),
            "loss": float(aggregated.loss),
        }

        metric_payload.update(
            {
                str(key): float(value)
                for key, value in aggregated.metrics.items()
            }
        )

        return MetricRecord(
            metric_payload,
        )

    # ==================================================================
    # Flower -> FedMed training result conversion
    # ==================================================================

    def _fit_result_from_message(
        self,
        message: Message,
    ) -> FederatedFitResult:
        """
        Convert a Flower NumPyClient compatibility FitRes into
        FederatedFitResult.

        Expected representation:

            fitres.parameters
            fitres.num_examples
            fitres.metrics
            fitres.status
        """

        content = message.content

        if FITRES_PARAMETERS_KEY not in content:
            raise FederatedLearningError(
                "Training reply is missing "
                f"'{FITRES_PARAMETERS_KEY}'."
            )

        if FITRES_NUM_EXAMPLES_KEY not in content:
            raise FederatedLearningError(
                "Training reply is missing "
                f"'{FITRES_NUM_EXAMPLES_KEY}'."
            )

        if FITRES_METRICS_KEY not in content:
            raise FederatedLearningError(
                "Training reply is missing "
                f"'{FITRES_METRICS_KEY}'."
            )

        # --------------------------------------------------------------
        # Parameters
        # --------------------------------------------------------------

        parameters_record = content[
            FITRES_PARAMETERS_KEY
        ]

        if not isinstance(
            parameters_record,
            ArrayRecord,
        ):
            raise FederatedLearningError(
                "Training reply "
                f"'{FITRES_PARAMETERS_KEY}' must be an "
                "ArrayRecord, "
                f"got {type(parameters_record).__name__}."
            )

        parameters = (
            parameters_record.to_numpy_ndarrays()
        )

        # --------------------------------------------------------------
        # Number of examples
        # --------------------------------------------------------------

        num_examples_record = content[
            FITRES_NUM_EXAMPLES_KEY
        ]

        if not isinstance(
            num_examples_record,
            MetricRecord,
        ):
            raise FederatedLearningError(
                "Training reply "
                f"'{FITRES_NUM_EXAMPLES_KEY}' must be a "
                "MetricRecord, "
                f"got {type(num_examples_record).__name__}."
            )

        if NUM_EXAMPLES_KEY not in num_examples_record:
            raise FederatedLearningError(
                "Training reply "
                f"'{FITRES_NUM_EXAMPLES_KEY}' is missing "
                f"'{NUM_EXAMPLES_KEY}'."
            )

        num_examples = self._extract_num_examples(
            num_examples_record,
        )

        # --------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------

        metrics_record = content[
            FITRES_METRICS_KEY
        ]

        if not isinstance(
            metrics_record,
            ConfigRecord,
        ):
            raise FederatedLearningError(
                "Training reply "
                f"'{FITRES_METRICS_KEY}' must be a "
                "ConfigRecord, "
                f"got {type(metrics_record).__name__}."
            )

        numeric_metrics = self._extract_numeric_metrics(
            metrics_record,
            exclude=set(),
        )

        return FederatedFitResult(
            parameters=self._copy_parameters(
                parameters,
            ),
            num_examples=num_examples,
            metrics=numeric_metrics,
            epochs_completed=int(
                numeric_metrics.get(
                    "epochs_completed",
                    0.0,
                )
            ),
            batches_processed=int(
                numeric_metrics.get(
                    "batches_processed",
                    0.0,
                )
            ),
            final_loss=float(
                numeric_metrics.get(
                    "final_loss",
                    numeric_metrics.get(
                        "train_loss",
                        0.0,
                    ),
                )
            ),
        )

    # ==================================================================
    # Flower -> FedMed evaluation result conversion
    # ==================================================================

    def _evaluate_result_from_message(
        self,
        message: Message,
    ) -> FederatedEvaluateResult:
        """
        Convert a Flower NumPyClient compatibility EvaluateRes into
        FederatedEvaluateResult.
        """

        content = message.content

        if EVALUATERES_LOSS_KEY not in content:
            raise FederatedLearningError(
                "Evaluation reply is missing "
                f"'{EVALUATERES_LOSS_KEY}'."
            )

        if EVALUATERES_NUM_EXAMPLES_KEY not in content:
            raise FederatedLearningError(
                "Evaluation reply is missing "
                f"'{EVALUATERES_NUM_EXAMPLES_KEY}'."
            )

        if EVALUATERES_METRICS_KEY not in content:
            raise FederatedLearningError(
                "Evaluation reply is missing "
                f"'{EVALUATERES_METRICS_KEY}'."
            )

        # --------------------------------------------------------------
        # Loss
        # --------------------------------------------------------------

        loss_record = content[
            EVALUATERES_LOSS_KEY
        ]

        if not isinstance(
            loss_record,
            MetricRecord,
        ):
            raise FederatedLearningError(
                "Evaluation reply "
                f"'{EVALUATERES_LOSS_KEY}' must be a "
                "MetricRecord."
            )

        if "loss" not in loss_record:
            raise FederatedLearningError(
                "Evaluation loss record is missing 'loss'."
            )

        loss = float(
            loss_record["loss"],
        )

        if not np.isfinite(loss):
            raise FederatedLearningError(
                "Evaluation loss must be finite."
            )

        # --------------------------------------------------------------
        # Number of examples
        # --------------------------------------------------------------

        num_examples_record = content[
            EVALUATERES_NUM_EXAMPLES_KEY
        ]

        if not isinstance(
            num_examples_record,
            MetricRecord,
        ):
            raise FederatedLearningError(
                "Evaluation reply "
                f"'{EVALUATERES_NUM_EXAMPLES_KEY}' must be a "
                "MetricRecord."
            )

        num_examples = self._extract_num_examples(
            num_examples_record,
        )

        # --------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------

        metrics_record = content[
            EVALUATERES_METRICS_KEY
        ]

        if not isinstance(
            metrics_record,
            ConfigRecord,
        ):
            raise FederatedLearningError(
                "Evaluation reply "
                f"'{EVALUATERES_METRICS_KEY}' must be a "
                "ConfigRecord."
            )

        metrics = self._extract_numeric_metrics(
            metrics_record,
            exclude=set(),
        )

        return FederatedEvaluateResult(
            num_examples=num_examples,
            loss=loss,
            metrics=metrics,
        )

    # ==================================================================
    # Node selection
    # ==================================================================

    def _select_nodes(
        self,
        grid: Grid,
        fraction: float,
    ) -> list[int]:
        """
        Select Flower nodes deterministically.

        The current FedMed baseline selects all available nodes when
        fraction == 1.0. Fractional selection is deterministic so
        runtime tests remain reproducible.
        """

        available = sorted(
            int(node_id)
            for node_id in grid.get_node_ids()
        )

        if len(available) < self._min_available_nodes:
            raise FederatedLearningError(
                "Insufficient Flower nodes available: "
                f"required at least "
                f"{self._min_available_nodes}, "
                f"found {len(available)}."
            )

        if fraction == 0.0:
            return []

        target_count = max(
            1,
            int(
                np.ceil(
                    len(available) * fraction,
                )
            ),
        )

        target_count = max(
            target_count,
            self._min_available_nodes,
        )

        if target_count > len(available):
            raise FederatedLearningError(
                "Insufficient Flower nodes available to satisfy "
                f"min_available_nodes="
                f"{self._min_available_nodes}."
            )

        return available[:target_count]

    # ==================================================================
    # Message construction
    # ==================================================================

    @staticmethod
    def _construct_messages(
        record: RecordDict,
        node_ids: Sequence[int],
        message_type: str,
    ) -> list[Message]:
        """
        Construct one Flower Message for each selected node.
        """

        return [
            Message(
                content=record,
                message_type=message_type,
                dst_node_id=node_id,
            )
            for node_id in node_ids
        ]

    # ==================================================================
    # Parameter utilities
    # ==================================================================

    @staticmethod
    def _copy_parameters(
        parameters: ParameterPayload,
    ) -> ParameterPayload:
        """
        Return a defensive copy of a parameter payload.
        """

        return [
            np.array(
                parameter,
                copy=True,
            )
            for parameter in parameters
        ]

    # ==================================================================
    # Metric utilities
    # ==================================================================

    @staticmethod
    def _extract_num_examples(
        metrics: Mapping[str, Any],
    ) -> int:
        """
        Extract and validate a positive sample count.
        """

        if NUM_EXAMPLES_KEY not in metrics:
            raise FederatedLearningError(
                "MetricRecord must contain "
                f"'{NUM_EXAMPLES_KEY}'."
            )

        value = metrics[
            NUM_EXAMPLES_KEY
        ]

        if isinstance(
            value,
            bool,
        ):
            raise FederatedLearningError(
                f"'{NUM_EXAMPLES_KEY}' must be an integer."
            )

        try:
            count = int(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise FederatedLearningError(
                f"'{NUM_EXAMPLES_KEY}' must be an integer."
            ) from exc

        if count <= 0:
            raise FederatedLearningError(
                f"'{NUM_EXAMPLES_KEY}' must be positive."
            )

        return count

    @staticmethod
    def _extract_numeric_metrics(
        metrics: Mapping[str, Any],
        *,
        exclude: set[str],
    ) -> dict[str, float]:
        """
        Extract finite numeric scalar metrics.

        Boolean values are intentionally excluded because bool is a
        subclass of int in Python but is not a meaningful training
        metric here.
        """

        normalized: dict[str, float] = {}

        for key, value in metrics.items():
            if key in exclude:
                continue

            if isinstance(
                value,
                bool,
            ):
                continue

            if isinstance(
                value,
                np.generic,
            ):
                value = value.item()

            if isinstance(
                value,
                (int, float),
            ):
                numeric = float(value)

                if not np.isfinite(numeric):
                    raise FederatedLearningError(
                        f"Metric '{key}' must be finite."
                    )

                normalized[str(key)] = numeric

        return normalized

    @staticmethod
    def _weighted_metrics(
        observations: Sequence[
            tuple[int, Mapping[str, float]]
        ],
    ) -> dict[str, float]:
        """
        Compute sample-weighted averages for common scalar metrics.
        """

        if not observations:
            raise FederatedLearningError(
                "Cannot aggregate metrics without observations."
            )

        total = sum(
            count
            for count, _ in observations
        )

        if total <= 0:
            raise FederatedLearningError(
                "Cannot aggregate metrics without positive "
                "sample counts."
            )

        keys: set[str] = set()

        for _, metrics in observations:
            keys.update(metrics.keys())

        result: dict[str, float] = {}

        for key in sorted(keys):
            weighted_sum = 0.0
            weight = 0

            for count, metrics in observations:
                if key not in metrics:
                    continue

                weighted_sum += (
                    metrics[key] * count
                )
                weight += count

            if weight:
                result[key] = (
                    weighted_sum / weight
                )

        result[
            NUM_EXAMPLES_KEY
        ] = float(total)

        return result

    # ==================================================================
    # Validation
    # ==================================================================

    @staticmethod
    def _validate_fraction(
        value: float,
        name: str,
    ) -> float:
        """
        Validate a node-selection fraction.
        """

        if isinstance(
            value,
            bool,
        ):
            raise FederatedLearningError(
                f"{name} must be a float in [0.0, 1.0]."
            )

        try:
            numeric = float(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise FederatedLearningError(
                f"{name} must be a float in [0.0, 1.0]."
            ) from exc

        if not 0.0 <= numeric <= 1.0:
            raise FederatedLearningError(
                f"{name} must be in [0.0, 1.0]."
            )

        return numeric

    # ==================================================================
    # Diagnostics
    # ==================================================================

    def summary(self) -> None:
        """Print the runtime strategy configuration."""

        print(
            "FedMedFlowerStrategy("
            f"fedmed_strategy="
            f"{type(self._fedmed_strategy).__name__}, "
            f"fraction_train="
            f"{self._fraction_train}, "
            f"fraction_evaluate="
            f"{self._fraction_evaluate}, "
            f"min_available_nodes="
            f"{self._min_available_nodes}"
            ")"
        )


# ======================================================================
# ServerApp factory
# ======================================================================


def create_server_app(
    initial_parameters_factory: InitialParametersFactory,
    strategy_factory: StrategyFactory | None = None,
    *,
    num_rounds: int = 1,
) -> ServerApp:
    """
    Create a Flower ServerApp around the FedMed runtime boundary.

    Model/data/client construction remains outside this adapter.

    ``initial_parameters_factory`` creates the initial global
    ParameterPayload.

    ``strategy_factory`` optionally creates the framework-independent
    FedMed Strategy.

    Flower owns execution of ``strategy.start()``.
    """

    if not callable(
        initial_parameters_factory,
    ):
        raise FederatedLearningError(
            "initial_parameters_factory must be callable."
        )

    if (
        strategy_factory is not None
        and not callable(strategy_factory)
    ):
        raise FederatedLearningError(
            "strategy_factory must be callable when supplied."
        )

    if (
        not isinstance(num_rounds, int)
        or isinstance(num_rounds, bool)
        or num_rounds < 1
    ):
        raise FederatedLearningError(
            "num_rounds must be a positive integer."
        )

    app = ServerApp()

    @app.main()
    def main(
        grid: Grid,
        context: Any,
    ) -> None:
        print(
            "[FedMed DEBUG] ServerApp main ENTER",
            flush=True,
        )

        # --------------------------------------------------------------
        # Initial global parameters
        # --------------------------------------------------------------

        parameters = initial_parameters_factory(
            context,
        )

        if not isinstance(
            parameters,
            list,
        ):
            raise FederatedLearningError(
                "initial_parameters_factory must return "
                "a ParameterPayload."
            )

        initial_arrays = ArrayRecord.from_numpy_ndarrays(
            FedMedFlowerStrategy._copy_parameters(
                parameters,
            )
        )

        # --------------------------------------------------------------
        # FedMed Strategy
        # --------------------------------------------------------------

        if strategy_factory is None:
            fedmed_strategy = FedAvgStrategy(
                aggregator=FedAvgAggregator(),
            )
        else:
            fedmed_strategy = strategy_factory(
                context,
            )

        if not isinstance(
            fedmed_strategy,
            FedAvgStrategy,
        ):
            raise FederatedLearningError(
                "strategy_factory must return a "
                "FedAvgStrategy."
            )

        # --------------------------------------------------------------
        # Flower Strategy adapter
        # --------------------------------------------------------------

        strategy = FedMedFlowerStrategy(
            fedmed_strategy=fedmed_strategy,
        )

        # --------------------------------------------------------------
        # Runtime configuration
        # --------------------------------------------------------------

        run_config = getattr(
            context,
            "run_config",
            {},
        )

        train_config = ConfigRecord(
            dict(run_config),
        )

        print(
            "[FedMed DEBUG] initial global parameters converted "
            "to ArrayRecord",
            flush=True,
        )

        print(
            "[FedMed DEBUG] starting Flower strategy for "
            f"{num_rounds} round(s)",
            flush=True,
        )

        # --------------------------------------------------------------
        # Flower strategy lifecycle
        # --------------------------------------------------------------

        strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=num_rounds,
            train_config=train_config,
        )

        print(
            "[FedMed DEBUG] Flower strategy.start returned",
            flush=True,
        )

    return app


__all__ = [
    "FedMedFlowerStrategy",
    "create_server_app",
    "FIT_PARAMETERS_KEY",
    "FIT_CONFIG_KEY",
    "FITRES_PARAMETERS_KEY",
    "FITRES_NUM_EXAMPLES_KEY",
    "FITRES_METRICS_KEY",
    "FITRES_STATUS_KEY",
    "EVALUATE_PARAMETERS_KEY",
    "EVALUATE_CONFIG_KEY",
    "EVALUATERES_LOSS_KEY",
    "EVALUATERES_NUM_EXAMPLES_KEY",
    "EVALUATERES_METRICS_KEY",
    "EVALUATERES_STATUS_KEY",
    "NUM_EXAMPLES_KEY",
]