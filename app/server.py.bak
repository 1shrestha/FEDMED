from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg as FlowerFedAvg

from src.common.exceptions import FederatedLearningError
from src.fl.client import FederatedEvaluateResult, FederatedFitResult
from src.fl.strategy import FedAvgEvaluationResult, FedAvgStrategy

ARRAYS_KEY = "arrays"
CONFIG_KEY = "config"
METRICS_KEY = "metrics"
NUM_EXAMPLES_KEY = "num_examples"


InitialParametersFactory = Callable[[Any], list[np.ndarray]]
StrategyFactory = Callable[[Any], FedAvgStrategy]


class FedMedFlowerStrategy(FlowerFedAvg):
    """
    Flower 1.34 server-side adapter.

    Flower handles:
        ClientApp communication
        ArrayRecord transport
        MetricRecord transport
        node sampling
        message lifecycle

    FedMed handles:
        FederatedClient
        FedAvgStrategy
        FedAvgAggregator
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

        if isinstance(fraction_train, bool) or not isinstance(
            fraction_train,
            (int, float),
        ):
            raise FederatedLearningError(
                "fraction_train must be a float between 0 and 1, "
                f"got {type(fraction_train).__name__}."
            )
        if not 0.0 < float(fraction_train) <= 1.0:
            raise FederatedLearningError(
                "fraction_train must be in the interval (0, 1], "
                f"got {fraction_train}."
            )

        if isinstance(fraction_evaluate, bool) or not isinstance(
            fraction_evaluate,
            (int, float),
        ):
            raise FederatedLearningError(
                "fraction_evaluate must be a float between 0 and 1, "
                f"got {type(fraction_evaluate).__name__}."
            )
        if not 0.0 < float(fraction_evaluate) <= 1.0:
            raise FederatedLearningError(
                "fraction_evaluate must be in the interval (0, 1], "
                f"got {fraction_evaluate}."
            )

        if isinstance(min_available_nodes, bool) or not isinstance(
            min_available_nodes,
            int,
        ):
            raise FederatedLearningError(
                "min_available_nodes must be an integer >= 1, "
                f"got {type(min_available_nodes).__name__}."
            )
        if min_available_nodes < 1:
            raise FederatedLearningError(
                "min_available_nodes must be >= 1, "
                f"got {min_available_nodes}."
            )

        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_available_nodes=min_available_nodes,
        )

        self._fedmed_strategy = fedmed_strategy

        print(
            "[FedMed] Flower strategy adapter created: "
            f"{type(self).__name__}"
        )

    @property
    def fedmed_strategy(self) -> FedAvgStrategy:
        return self._fedmed_strategy

    def _select_nodes(
        self,
        grid: Grid,
        num_nodes: float,
    ) -> list[int]:
        available = sorted(grid.get_node_ids())
        if not available:
            raise FederatedLearningError("Insufficient Flower nodes available.")

        target_count = max(
            1,
            min(
                len(available),
                int(np.ceil(len(available) * float(num_nodes))),
            ),
        )
        target_count = max(
            target_count,
            self.min_available_nodes,
        )
        if target_count > len(available):
            raise FederatedLearningError(
                "Insufficient Flower nodes available to satisfy "
                f"min_available_nodes={self.min_available_nodes}."
            )

        return available[:target_count]

    def configure_train(
        self,
        *,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        selected_nodes = self._select_nodes(grid, self.fraction_train)
        for node_id in selected_nodes:
            node_config = ConfigRecord(dict(config))
            node_config["server-round"] = server_round
            yield Message(
                content=RecordDict(
                    {
                        ARRAYS_KEY: arrays,
                        CONFIG_KEY: node_config,
                    }
                ),
                dst_node_id=node_id,
                message_type="train",
            )

    def configure_evaluate(
        self,
        *,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        selected_nodes = self._select_nodes(grid, self.fraction_evaluate)
        for node_id in selected_nodes:
            node_config = ConfigRecord(dict(config))
            node_config["server-round"] = server_round
            yield Message(
                content=RecordDict(
                    {
                        ARRAYS_KEY: arrays,
                        CONFIG_KEY: node_config,
                    }
                ),
                dst_node_id=node_id,
                message_type="evaluate",
            )

    @staticmethod
    def _copy_parameters(
        parameters: list[np.ndarray],
    ) -> list[np.ndarray]:
        return [np.array(parameter, copy=True) for parameter in parameters]

    def _fit_result_from_message(
        self,
        message: Message,
    ) -> FederatedFitResult:
        if ARRAYS_KEY not in message.content:
            raise FederatedLearningError(
                "training reply missing 'arrays' payload."
            )
        if METRICS_KEY not in message.content:
            raise FederatedLearningError(
                "training reply missing 'metrics' payload."
            )

        arrays = message.content[ARRAYS_KEY]
        metrics = dict(message.content[METRICS_KEY])

        if NUM_EXAMPLES_KEY not in metrics:
            raise FederatedLearningError(
                "training reply missing 'num_examples' metric."
            )

        parameters = arrays.to_numpy_ndarrays()
        num_examples = int(metrics[NUM_EXAMPLES_KEY])
        metric_payload = {
            str(key): float(value)
            for key, value in metrics.items()
            if key != NUM_EXAMPLES_KEY
        }

        return FederatedFitResult(
            parameters=self._copy_parameters(parameters),
            num_examples=num_examples,
            metrics=metric_payload,
            epochs_completed=int(metrics.get("epochs_completed", 0)),
            batches_processed=int(metrics.get("batches_processed", 0)),
            final_loss=float(metrics.get("final_loss", 0.0)),
        )

    def _evaluate_result_from_message(
        self,
        message: Message,
    ) -> FederatedEvaluateResult:
        if METRICS_KEY not in message.content:
            raise FederatedLearningError(
                "evaluation reply missing 'metrics' payload."
            )

        metrics = dict(message.content[METRICS_KEY])
        if NUM_EXAMPLES_KEY not in metrics:
            raise FederatedLearningError(
                "evaluation reply missing 'num_examples' metric."
            )
        if "loss" not in metrics:
            raise FederatedLearningError(
                "evaluation reply missing 'loss' metric."
            )

        metric_payload = {
            str(key): float(value)
            for key, value in metrics.items()
            if key not in {NUM_EXAMPLES_KEY, "loss"}
        }

        return FederatedEvaluateResult(
            num_examples=int(metrics[NUM_EXAMPLES_KEY]),
            loss=float(metrics["loss"]),
            metrics=metric_payload,
        )

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies = list(replies)
        if not replies:
            return None, None

        results: dict[str, FederatedFitResult] = {}
        seen_nodes: set[int] = set()

        for reply in replies:
            node_id = int(reply.metadata.src_node_id)
            if node_id in seen_nodes:
                raise FederatedLearningError(
                    f"Duplicate training reply from node {node_id}."
                )
            seen_nodes.add(node_id)
            result = self._fit_result_from_message(reply)
            results[str(node_id)] = result

        print(
            "[FedMed] Delegating Flower training aggregation to "
            f"FedAvgAggregator for round {server_round}."
        )
        aggregated_parameters = self._fedmed_strategy.aggregate_fit(
            results,
            round_number=server_round,
        )
        total_examples = sum(
            result.num_examples for result in results.values()
        )
        metrics = MetricRecord({NUM_EXAMPLES_KEY: float(total_examples)})
        return ArrayRecord.from_numpy_ndarrays(aggregated_parameters), metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        replies = list(replies)
        if not replies:
            return None

        results: dict[str, FederatedEvaluateResult] = {}
        seen_nodes: set[int] = set()

        for reply in replies:
            node_id = int(reply.metadata.src_node_id)
            if node_id in seen_nodes:
                raise FederatedLearningError(
                    f"Duplicate evaluation reply from node {node_id}."
                )
            seen_nodes.add(node_id)
            result = self._evaluate_result_from_message(reply)
            results[str(node_id)] = result

        print(
            "[FedMed] Delegating Flower evaluation aggregation to "
            f"FedAvgAggregator for round {server_round}."
        )
        aggregated: FedAvgEvaluationResult = (
            self._fedmed_strategy.aggregate_evaluate(
                results,
                round_number=server_round,
            )
        )
        total_examples = sum(
            result.num_examples for result in results.values()
        )
        metric_payload: dict[str, float | int] = {
            NUM_EXAMPLES_KEY: float(total_examples),
            "loss": float(aggregated.loss),
        }
        metric_payload.update(
            {
                str(key): float(value)
                for key, value in aggregated.metrics.items()
            }
        )
        return MetricRecord(metric_payload)


def create_server_app(
    initial_parameters_factory: InitialParametersFactory,
    strategy_factory: StrategyFactory | None = None,
    num_rounds: int = 1,
) -> ServerApp:
    """
    Create the Flower ServerApp.

    Flower owns the runtime lifecycle.
    FedMed owns its framework-independent federation logic.
    """

    if not callable(initial_parameters_factory):
        raise FederatedLearningError(
            "initial_parameters_factory must be callable."
        )

    if not isinstance(num_rounds, int) or isinstance(num_rounds, bool):
        raise FederatedLearningError(
            "num_rounds must be a positive integer."
        )

    if num_rounds < 1:
        raise FederatedLearningError(
            "num_rounds must be a positive integer."
        )

    if strategy_factory is None:
        strategy_factory = (
            lambda context: FedAvgStrategy(
                aggregator=FedAvgAggregator(),
            )
        )
    elif not callable(strategy_factory):
        raise FederatedLearningError(
            "strategy_factory must be callable."
        )

    app = ServerApp()

    @app.main()
    def main(
        grid: Grid,
        context: Context,
    ) -> None:
        print("=" * 70)
        print("FedMed — FLOWER SERVER APP")
        print("=" * 70)

        # --------------------------------------------------------
        # BUILD EXISTING FEDMED STRATEGY
        # --------------------------------------------------------

        fedmed_strategy = strategy_factory(context)

        print(
            "[FedMed] FedMed strategy: "
            f"{type(fedmed_strategy).__name__}"
        )

        # --------------------------------------------------------
        # BUILD FLOWER ADAPTER
        # --------------------------------------------------------

        flower_strategy = FedMedFlowerStrategy(
            fedmed_strategy=fedmed_strategy,
        )

        print(
            "[FedMed] Flower adapter: "
            f"{type(flower_strategy).__name__}"
        )

        # --------------------------------------------------------
        # INITIAL PARAMETERS
        # --------------------------------------------------------

        initial_parameters = initial_parameters_factory(
            context
        )

        initial_arrays = ArrayRecord.from_numpy_ndarrays(
            initial_parameters
        )

        print(
            "[FedMed] Initial parameter tensors: "
            f"{len(initial_parameters)}"
        )

        for index, parameter in enumerate(initial_parameters):
            print(
                f"  [{index}] "
                f"shape={parameter.shape}, "
                f"dtype={parameter.dtype}"
            )

        # --------------------------------------------------------
        # FLOWER FEDERATED EXECUTION
        # --------------------------------------------------------

        print()
        print(
            "[FedMed] Starting Flower strategy..."
        )

        result = flower_strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            train_config=ConfigRecord(),
            num_rounds=num_rounds,
        )

        # --------------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------------

        final_parameters = (
            result.arrays.to_numpy_ndarrays()
        )

        print()
        print("=" * 70)
        print("FedMed — FLOWER FEDERATION COMPLETE")
        print("=" * 70)

        print(
            "Final parameter tensors: "
            f"{len(final_parameters)}"
        )

        for index, parameter in enumerate(final_parameters):
            print(
                f"  [{index}] "
                f"shape={parameter.shape}, "
                f"dtype={parameter.dtype}"
            )

        print()
        print(
            "[FedMed] Flower training metrics:"
        )

        for round_number, metrics in (
            result.train_metrics_clientapp.items()
        ):
            print(
                f"  Round {round_number}: "
                f"{dict(metrics)}"
            )

        print("=" * 70)


    return app


__all__ = [
    "FedMedFlowerStrategy",
    "create_server_app",
]