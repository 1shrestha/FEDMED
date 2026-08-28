"""
FedMed application composition root.

This module is the single place where the framework-independent FedMed
components are assembled and handed to the Flower adapters.

Dependency flow
---------------

Client side:
    Model -> Trainer
          -> Evaluator
          -> FederatedClient -> app.client.create_client_app

Server side:
    FedAvgAggregator -> FedAvgStrategy -> app.server.create_server_app

The Flower adapters remain thin transport/runtime boundaries. This module
owns construction; it does not duplicate training or aggregation logic.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from app.client import create_client_app
from app.server import create_server_app
from src.aggregation.fedavg import FedAvgAggregator
from src.common.config import TrainingConfig
from src.fl.client import FederatedClient
from src.fl.strategy import FedAvgStrategy
from src.models.base_model import BaseModel
from src.training.evaluator import Evaluator
from src.training.metrics import Accuracy
from src.training.trainer import Trainer


class FlowerSmokeTestModel(BaseModel):
    """Small deterministic model used by the Flower integration runtime."""

    def build(self) -> nn.Module:
        return nn.Linear(2, 2)


class FedMedOrchestrator:
    """Central composition root for the FedMed Flower application."""

    def __init__(self) -> None:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        self._training_config = TrainingConfig(
            local_epochs=1,
            batch_size=4,
            learning_rate=0.01,
            optimizer="sgd",
            seed=42,
        )

    # ------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------

    @staticmethod
    def _create_loader(client_id: str, *, size: int = 8) -> DataLoader:
        """Create deterministic local smoke-test data for one client."""
        if size < 2 or size % 2 != 0:
            raise ValueError("size must be an even integer >= 2")

        client_offset = sum(ord(c) for c in client_id) % 10

        generator = torch.Generator()
        generator.manual_seed(42 + client_offset)

        samples = torch.randn(size, 2, generator=generator)
        targets = torch.tensor([0, 1] * (size // 2), dtype=torch.long)

        return DataLoader(
            TensorDataset(samples, targets),
            batch_size=4,
            shuffle=False,
        )

    # ------------------------------------------------------------------
    # CLIENT ASSEMBLY
    # ------------------------------------------------------------------

    def build_client(self, client_id: str) -> FederatedClient:
        """Build one complete FedMed client dependency graph."""
        if not isinstance(client_id, str) or not client_id.strip():
            raise ValueError("client_id must be a non-empty string")

        torch.manual_seed(100)

        model = FlowerSmokeTestModel(
            name=f"flower_smoke_{client_id}",
            device="cpu",
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(
            model.parameters(),
            lr=self._training_config.learning_rate,
        )

        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            config=self._training_config,
        )

        evaluator = Evaluator(
            model=model,
            criterion=criterion,
            metrics=[Accuracy()],
        )

        train_loader = self._create_loader(client_id, size=8)
        eval_loader = self._create_loader(f"{client_id}_eval", size=8)

        client = FederatedClient(
            client_id=client_id,
            model=model,
            trainer=trainer,
            evaluator=evaluator,
            train_loader=train_loader,
            eval_loader=eval_loader,
        )

        print(f"[FedMed] client assembled: {client_id}")
        return client

    # ------------------------------------------------------------------
    # SERVER STRATEGY ASSEMBLY
    # ------------------------------------------------------------------

    @staticmethod
    def build_strategy() -> FedAvgStrategy:
        """Build Strategy -> Aggregator without implementing FedAvg here."""
        aggregator = FedAvgAggregator()
        strategy = FedAvgStrategy(aggregator=aggregator)

        print(
            "[FedMed] strategy assembled: "
            f"{type(strategy).__name__} -> {type(aggregator).__name__}"
        )
        return strategy

    # ------------------------------------------------------------------
    # FLOWER CLIENT APP
    # ------------------------------------------------------------------

    def build_client_app(self):
        """Build the Flower ClientApp using the central client factory."""

        def client_factory(context: Any) -> FederatedClient:
            node_id = str(context.node_id)
            return self.build_client(f"client_{node_id}")

        return create_client_app(client_factory)

    # ------------------------------------------------------------------
    # FLOWER SERVER APP
    # ------------------------------------------------------------------

    def build_server_app(self):
        """Build the Flower ServerApp using central server factories."""

        def initial_parameters_factory(context: Any):
            del context
            # The same assembly path used by a real client creates the
            # canonical model parameter payload for the initial global model.
            initial_client = self.build_client("initial")
            parameters = initial_client.get_parameters()
            print("[FedMed] initial global parameters created")
            return parameters

        def strategy_factory(context: Any):
            del context
            return self.build_strategy()

        return create_server_app(
            initial_parameters_factory,
            strategy_factory=strategy_factory,
            num_rounds=1,
        )

    # ------------------------------------------------------------------
    # COMPLETE APPLICATION
    # ------------------------------------------------------------------

    def build_apps(self):
        """Assemble and return ``(client_app, server_app)``."""
        print("[FedMed] assembling Flower application")

        client_app = self.build_client_app()
        server_app = self.build_server_app()

        print("[FedMed] Flower ClientApp assembled")
        print("[FedMed] Flower ServerApp assembled")

        return client_app, server_app


__all__ = ["FedMedOrchestrator", "FlowerSmokeTestModel"]
