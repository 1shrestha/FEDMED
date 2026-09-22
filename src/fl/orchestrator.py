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
from src.data.loader import create_dataloader

from app.client import create_client_app
from app.server import create_server_app
from src.aggregation.fedavg import FedAvgAggregator
from src.common.config import load_config
from src.data.dataset import FedMedDataset
from src.data.partitioner import PartitionView, partition_dataset
from src.fl.client import FederatedClient
from src.fl.strategy import FedAvgStrategy
from src.models.base_model import BaseModel
from src.training.evaluator import Evaluator
from src.training.metrics import Accuracy
from src.training.trainer import Trainer


# PyTorch inter-op thread configuration is process-wide and can only be
# changed before parallel work starts. Configure it once when this module
# is loaded rather than every time an orchestrator is constructed.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class FlowerSmokeTestModel(BaseModel):
    """Small deterministic model used by the Flower integration runtime."""

    def build(self) -> nn.Module:
        return nn.Linear(2, 2)


class FedMedOrchestrator:
    """Central composition root for the FedMed Flower application."""

    def __init__(self) -> None:
        config = load_config()
        self._training_config = config.training
        self._data_config = config.data

    # ------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------

    def _create_partitioned_loader(
        self,
        partition_index: int,
        *,
        seed: int = 42,
        split: str = "train",
    ) -> DataLoader:
        """Create a deterministic DataLoader for a Flower experiment split."""

        num_clients = self._data_config.num_clients

        if partition_index < 0 or partition_index >= num_clients:
            raise ValueError(
                f"partition index {partition_index} is outside "
                f"configured range 0..{num_clients - 1}"
            )

        if split not in {"train", "eval"}:
            raise ValueError(
                f"split must be 'train' or 'eval', got {split!r}"
            )

        if num_clients != 4:
            raise ValueError(
                "E8 data imbalance requires exactly 4 configured clients."
            )

        # E8 intentionally varies training-data quantity while keeping
        # evaluation-data quantity balanced.
        sizes = (4, 8, 8, 12) if split == "train" else (8, 8, 8, 8)
        size = sum(sizes)

        generator = torch.Generator()
        generator.manual_seed(seed)

        samples = torch.randn(size, 2, generator=generator)
        targets = torch.tensor(
            [0, 1] * (size // 2),
            dtype=torch.long,
        )

        dataset = FedMedDataset(
            samples=samples,
            targets=targets,
            name=f"flower_smoke_{split}_global",
        )

        if self._data_config.partition_type == "iid":
            start = sum(sizes[:partition_index])
            end = start + sizes[partition_index]
            indices = tuple(range(start, end))
        else:
            partitions = partition_dataset(
                dataset,
                num_clients=num_clients,
                strategy=self._data_config.partition_type,
                seed=seed,
            )
            indices = partitions[f"client_{partition_index}"].indices

        partition = PartitionView(
            dataset=dataset,
            indices=indices,
            client_id=f"client_{partition_index}",
        )

        print(
            f"[FedMed] partition assembled: "
            f"client_{partition_index} "
            f"split={split} "
            f"strategy={self._data_config.partition_type} "
            f"samples={len(partition)}"
        )

        return create_dataloader(
            partition,
            batch_size=self._training_config.batch_size,
            shuffle=False,
        )

    # ------------------------------------------------------------------
    # CLIENT ASSEMBLY
    # ------------------------------------------------------------------

    def build_client(
        self,
        client_id: str,
        *,
        partition_index: int | None = None,
    ) -> FederatedClient:
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

        if client_id == "initial":
            partition_index = 0
        elif partition_index is None:
            try:
                node_id = int(client_id.rsplit("_", 1)[1])
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"client_id must end with a numeric node id: {client_id!r}"
                ) from exc

            partition_index = node_id % self._data_config.num_clients

        train_loader = self._create_partitioned_loader(partition_index)
        eval_loader = self._create_partitioned_loader(
            partition_index,
            split="eval",
        )

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

            partition_index = int(
                context.node_config["partition-id"]
            )

            return self.build_client(
                f"client_{node_id}",
                partition_index=partition_index,
            )

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
