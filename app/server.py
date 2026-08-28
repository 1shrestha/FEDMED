from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg as FlowerFedAvg

from src.fl.strategy import FedAvgStrategy


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
    ) -> None:
        super().__init__(
            fraction_train=1.0,
            fraction_evaluate=1.0,
        )

        self._fedmed_strategy = fedmed_strategy

        print(
            "[FedMed] Flower strategy adapter created: "
            f"{type(self).__name__}"
        )

    @property
    def fedmed_strategy(self) -> FedAvgStrategy:
        return self._fedmed_strategy


def create_server_app(
    initial_parameters_factory: InitialParametersFactory,
    strategy_factory: StrategyFactory,
    num_rounds: int = 1,
) -> ServerApp:
    """
    Create the Flower ServerApp.

    Flower owns the runtime lifecycle.
    FedMed owns its framework-independent federation logic.
    """

    if not callable(initial_parameters_factory):
        raise TypeError(
            "initial_parameters_factory must be callable."
        )

    if not callable(strategy_factory):
        raise TypeError(
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