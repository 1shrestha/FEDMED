"""
FedMed Flower application assembly.

Phase 1 runtime smoke test
--------------------------

This module wires the existing FedMed Flower adapters to a small
deterministic synthetic client setup.

The synthetic model/data exist only to verify the Flower runtime
integration. They are not part of the FedMed production ML layer.

Architecture:

    Synthetic local data
            |
        Trainer
            |
        Evaluator
            |
    FederatedClient
            |
    FedMedNumPyClient
            |
        ClientApp
            |
       Flower Runtime
            |
        ServerApp
            |
    FedMedFlowerStrategy
            |
      FedAvgStrategy
            |
     FedAvgAggregator
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from app.client import create_client_app
from app.server import create_server_app
from src.common.config import TrainingConfig
from src.fl.client import FederatedClient
from src.models.base_model import BaseModel
from src.training.evaluator import Evaluator
from src.training.metrics import Accuracy
from src.training.trainer import Trainer


# ======================================================================
# Phase 1 smoke-test model
# ======================================================================


class FlowerSmokeTestModel(BaseModel):
    """Small deterministic model used only for Flower integration."""

    def build(self) -> nn.Module:
        return nn.Linear(2, 2)


# ======================================================================
# Synthetic local data
# ======================================================================


def _make_loader(
    client_id: str,
    *,
    size: int = 8,
) -> DataLoader:
    """
    Create deterministic synthetic local data.

    Each client receives a deterministic but slightly different
    dataset so that the Flower round performs genuine local training.
    """

    # Derive a stable small offset from the client identifier.
    client_offset = sum(
        ord(character)
        for character in client_id
    ) % 10

    generator = torch.Generator()
    generator.manual_seed(42 + client_offset)

    samples = torch.randn(
        size,
        2,
        generator=generator,
    )

    targets = torch.tensor(
        [0, 1] * (size // 2),
        dtype=torch.long,
    )

    return DataLoader(
        TensorDataset(
            samples,
            targets,
        ),
        batch_size=4,
        shuffle=False,
    )


# ======================================================================
# Federated client construction
# ======================================================================


def _make_federated_client(
    client_id: str,
) -> FederatedClient:
    """
    Construct one complete FedMed FederatedClient.

    This intentionally uses the existing Phase 2 Trainer/Evaluator
    and the existing Phase 3 FederatedClient contract.
    """

    torch.manual_seed(100)

    model = FlowerSmokeTestModel(
        name=f"flower_smoke_{client_id}",
        device="cpu",
    )

    criterion = nn.CrossEntropyLoss()

    training_config = TrainingConfig(
        local_epochs=1,
        batch_size=4,
        learning_rate=0.01,
        optimizer="sgd",
        seed=42,
    )

    optimizer = SGD(
        model.parameters(),
        lr=training_config.learning_rate,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        config=training_config,
    )

    evaluator = Evaluator(
        model=model,
        criterion=criterion,
        metrics=[Accuracy()],
    )

    train_loader = _make_loader(
        client_id,
        size=8,
    )

    eval_loader = _make_loader(
        f"{client_id}_eval",
        size=8,
    )

    return FederatedClient(
        client_id=client_id,
        model=model,
        trainer=trainer,
        evaluator=evaluator,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )


# ======================================================================
# Flower application factories
# ======================================================================


def client_factory(context: Any) -> FederatedClient:
    """
    Build the FedMed client requested by Flower.

    Flower provides the node identity through Context.
    """

    node_id = str(context.node_id)

    return _make_federated_client(
        f"client_{node_id}",
    )


def initial_parameters_factory(
    context: Any,
):
    """
    Build the initial global parameter payload.

    The initial parameters come from a valid FedMed client so that
    the parameter structure is guaranteed to match the client model.
    """

    initial_client = _make_federated_client(
        "initial",
    )

    return initial_client.get_parameters()


# ======================================================================
# Flower applications
# ======================================================================


client_app = create_client_app(
    client_factory,
)


server_app = create_server_app(
    initial_parameters_factory,
    num_rounds=1,
)


__all__ = [
    "client_app",
    "server_app",
]