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
    Flower Messages / ArrayRecords

The existing FedMed RoundCoordinator and FederatedServer remain the
framework-independent/in-process execution path. This module does
not create fake FederatedClient instances for remote Flower nodes.
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
from flwr.serverapp import ServerApp
from flwr.serverapp.strategy import Strategy

from src.aggregation.fedavg import FedAvgAggregator
from src.common.exceptions import FederatedLearningError
from src.fl.client import FederatedFitResult
from src.fl.parameters import ParameterPayload
from src.fl.strategy import FedAvgStrategy, FederatedStrategy


ARRAYS_KEY = "arrays"
CONFIG_KEY = "config"
METRICS_KEY = "metrics"
NUM_EXAMPLES_KEY = "num-examples"

InitialParametersFactory = Callable[[Any], ParameterPayload]
StrategyFactory = Callable[[Any], FederatedStrategy]


class FedMedFlowerStrategy(Strategy):
    """Flower Strategy adapter backed by FedMed aggregation contracts.

    The current FedAvgStrategy selects all available FedMed clients,
    while Flower's distributed server exposes node IDs instead. For
    this baseline adapter, deterministic node selection is therefore
    performed at the transport boundary. Parameter aggregation is
    still delegated to the injected FedMed Strategy/Aggregator.

    This avoids manufacturing fake data-owning FederatedClient objects.
    """

    def __init__(
        self,
        fedmed_strategy: FederatedStrategy,
        *,
        min_available_nodes: int = 1,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        arrayrecord_key: str = ARRAYS_KEY,
        configrecord_key: str = CONFIG_KEY,
        metricsrecord_key: str = METRICS_KEY,
    ) -> None:
        if not isinstance(fedmed_strategy, FederatedStrategy):
            raise FederatedLearningError(
                "fedmed_strategy must be a FederatedStrategy."
            )

        if (
            not isinstance(min_available_nodes, int)
            or isinstance(min_available_nodes, bool)
            or min_available_nodes < 1
        ):
            raise FederatedLearningError(
                "min_available_nodes must be a positive integer."
            )

        self._fedmed_strategy = fedmed_strategy
        self._min_available_nodes = min_available_nodes
        self._fraction_train = self._validate_fraction(
            fraction_train, "fraction_train"
        )
        self._fraction_evaluate = self._validate_fraction(
            fraction_evaluate, "fraction_evaluate"
        )
        self._arrayrecord_key = self._validate_key(
            arrayrecord_key, "arrayrecord_key"
        )
        self._configrecord_key = self._validate_key(
            configrecord_key, "configrecord_key"
        )
        self._metricsrecord_key = self._validate_key(
            metricsrecord_key, "metricsrecord_key"
        )

    @property
    def fedmed_strategy(self) -> FederatedStrategy:
        """Return the injected framework-independent strategy."""
        return self._fedmed_strategy

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Any,
    ) -> Iterable[Message]:
        """Construct Flower TRAIN messages for selected nodes."""

        node_ids = self._select_nodes(
            grid,
            self._fraction_train,
        )

        config = ConfigRecord(config)
        config["server-round"] = server_round

        record = RecordDict(
            {
                self._arrayrecord_key: arrays,
                self._configrecord_key: config,
            }
        )

        return self._construct_messages(
            record,
            node_ids,
            MessageType.TRAIN,
        )

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Convert Flower replies and delegate parameter aggregation."""

        results: dict[str, FederatedFitResult] = {}

        for message in replies:
            if message.has_error():
                continue

            client_id = str(message.metadata.src_node_id)

            if client_id in results:
                raise FederatedLearningError(
                    f"Duplicate training reply for client '{client_id}'."
                )

            results[client_id] = self._fit_result_from_message(message)

        if not results:
            return None, None

        aggregated = self._fedmed_strategy.aggregate_fit(
            results,
            server_round,
        )

        arrays = ArrayRecord.from_numpy_ndarrays(
            self._copy_parameters(aggregated)
        )

        metrics = MetricRecord(
            self._weighted_metrics(
                [
                    (result.num_examples, result.metrics)
                    for result in results.values()
                ]
            )
        )

        return arrays, metrics

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Any,
    ) -> Iterable[Message]:
        """Construct Flower EVALUATE messages for selected nodes."""

        node_ids = self._select_nodes(
            grid,
            self._fraction_evaluate,
        )

        config = ConfigRecord(config)
        config["server-round"] = server_round

        record = RecordDict(
            {
                self._arrayrecord_key: arrays,
                self._configrecord_key: config,
            }
        )

        return self._construct_messages(
            record,
            node_ids,
            MessageType.EVALUATE,
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        """Aggregate evaluation metrics without aggregating parameters."""

        del server_round

        observations: list[tuple[int, Mapping[str, float]]] = []

        for message in replies:
            if message.has_error():
                continue

            content = message.content

            if self._metricsrecord_key not in content:
                continue

            metric_record = content[self._metricsrecord_key]

            count = self._extract_num_examples(metric_record)
            metrics = self._extract_numeric_metrics(
                metric_record,
                exclude={NUM_EXAMPLES_KEY},
            )

            observations.append((count, metrics))

        if not observations:
            return None

        return MetricRecord(
            self._weighted_metrics(observations)
        )

    def summary(self) -> None:
        """Print the runtime strategy configuration."""
        print(
            "FedMedFlowerStrategy("
            f"fedmed_strategy={type(self._fedmed_strategy).__name__}, "
            f"fraction_train={self._fraction_train}, "
            f"fraction_evaluate={self._fraction_evaluate}, "
            f"min_available_nodes={self._min_available_nodes}"
            ")"
        )

    def _select_nodes(
        self,
        grid: Any,
        fraction: float,
    ) -> list[int]:
        """Select Flower node IDs deterministically for the baseline."""

        node_ids = sorted(
            int(node_id)
            for node_id in grid.get_node_ids()
        )

        if len(node_ids) < self._min_available_nodes:
            raise FederatedLearningError(
                "Insufficient Flower nodes: required at least "
                f"{self._min_available_nodes}, found {len(node_ids)}."
            )

        if fraction == 0.0:
            return []

        sample_size = max(
            int(len(node_ids) * fraction),
            self._min_available_nodes,
        )

        return node_ids[: min(sample_size, len(node_ids))]

    @staticmethod
    def _construct_messages(
        record: RecordDict,
        node_ids: Sequence[int],
        message_type: str,
    ) -> list[Message]:
        """Create one Flower message per selected node."""

        return [
            Message(
                content=record,
                message_type=message_type,
                dst_node_id=node_id,
            )
            for node_id in node_ids
        ]

    def _fit_result_from_message(
        self,
        message: Message,
    ) -> FederatedFitResult:
        """Convert a successful Flower training reply."""

        content = message.content

        if self._arrayrecord_key not in content:
            raise FederatedLearningError(
                f"Training reply is missing '{self._arrayrecord_key}'."
            )

        if self._metricsrecord_key not in content:
            raise FederatedLearningError(
                f"Training reply is missing '{self._metricsrecord_key}'."
            )

        arrays = content[self._arrayrecord_key]
        metrics = content[self._metricsrecord_key]

        parameters = arrays.to_numpy_ndarrays()
        num_examples = self._extract_num_examples(metrics)

        numeric_metrics = self._extract_numeric_metrics(
            metrics,
            exclude={NUM_EXAMPLES_KEY},
        )

        return FederatedFitResult(
            parameters=self._copy_parameters(parameters),
            num_examples=num_examples,
            metrics=numeric_metrics,
            epochs_completed=int(
                numeric_metrics.get("epochs_completed", 0.0)
            ),
            batches_processed=int(
                numeric_metrics.get("batches_processed", 0.0)
            ),
            final_loss=float(
                numeric_metrics.get(
                    "final_loss",
                    numeric_metrics.get("train_loss", 0.0),
                )
            ),
        )

    @staticmethod
    def _extract_num_examples(
        metrics: Mapping[str, Any],
    ) -> int:
        """Extract a positive local sample count."""

        if NUM_EXAMPLES_KEY not in metrics:
            raise FederatedLearningError(
                f"MetricRecord must contain '{NUM_EXAMPLES_KEY}'."
            )

        value = metrics[NUM_EXAMPLES_KEY]

        if isinstance(value, bool):
            raise FederatedLearningError(
                f"'{NUM_EXAMPLES_KEY}' must be an integer."
            )

        try:
            count = int(value)
        except (TypeError, ValueError) as exc:
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
        """Extract finite scalar numeric metrics."""

        normalized: dict[str, float] = {}

        for key, value in metrics.items():
            if key in exclude or isinstance(value, bool):
                continue

            if isinstance(value, np.generic):
                value = value.item()

            if isinstance(value, (int, float)):
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
        """Compute sample-weighted scalar metric averages."""

        total = sum(
            count
            for count, _ in observations
        )

        if total <= 0:
            raise FederatedLearningError(
                "Cannot aggregate metrics without positive sample counts."
            )

        keys: set[str] = set()
        for _, metrics in observations:
            keys.update(metrics)

        result: dict[str, float] = {}

        for key in sorted(keys):
            weighted_sum = 0.0
            weight = 0

            for count, metrics in observations:
                if key in metrics:
                    weighted_sum += (
                        metrics[key] * count
                    )
                    weight += count

            if weight:
                result[key] = weighted_sum / weight

        result[NUM_EXAMPLES_KEY] = float(total)

        return result

    @staticmethod
    def _copy_parameters(
        parameters: ParameterPayload,
    ) -> ParameterPayload:
        """Defensively copy parameter arrays."""

        return [
            np.array(parameter, copy=True)
            for parameter in parameters
        ]

    @staticmethod
    def _validate_fraction(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool):
            raise FederatedLearningError(
                f"{name} must be a float in [0.0, 1.0]."
            )

        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise FederatedLearningError(
                f"{name} must be a float in [0.0, 1.0]."
            ) from exc

        if not 0.0 <= numeric <= 1.0:
            raise FederatedLearningError(
                f"{name} must be in [0.0, 1.0]."
            )

        return numeric

    @staticmethod
    def _validate_key(
        value: str,
        name: str,
    ) -> str:
        if not isinstance(value, str) or not value.strip():
            raise FederatedLearningError(
                f"{name} must be a non-empty string."
            )

        return value


def create_server_app(
    initial_parameters_factory: InitialParametersFactory,
    strategy_factory: StrategyFactory | None = None,
    *,
    num_rounds: int = 1,
) -> ServerApp:
    """Create a Flower ServerApp around the FedMed runtime boundary.

    Model/data/client construction is intentionally not performed here.
    ``initial_parameters_factory`` and ``strategy_factory`` are
    application-assembly hooks and will be wired to configuration in
    the next phase.
    """

    if not callable(initial_parameters_factory):
        raise FederatedLearningError(
            "initial_parameters_factory must be callable."
        )

    if strategy_factory is not None and not callable(
        strategy_factory
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
    def main(grid: Any, context: Any) -> None:
        parameters = initial_parameters_factory(context)

        if strategy_factory is None:
            fedmed_strategy = FedAvgStrategy(
                aggregator=FedAvgAggregator()
            )
        else:
            fedmed_strategy = strategy_factory(context)

        strategy = FedMedFlowerStrategy(
            fedmed_strategy=fedmed_strategy
        )

        initial_arrays = ArrayRecord.from_numpy_ndarrays(
            FedMedFlowerStrategy._copy_parameters(parameters)
        )

        run_config = getattr(
            context,
            "run_config",
            {},
        )

        train_config = ConfigRecord(
            dict(run_config)
        )

        strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=num_rounds,
            train_config=train_config,
        )

    return app


__all__ = [
    "FedMedFlowerStrategy",
    "create_server_app",
]