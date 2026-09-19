from dataclasses import replace

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from src.common.config import load_config
from src.data.dataset import FedMedDataset
from src.fl.orchestrator import FlowerSmokeTestModel
from src.training.evaluator import Evaluator
from src.training.metrics import Accuracy
from src.training.trainer import Trainer


def test_e9_centralized_training():
    config = load_config()

    generator = torch.Generator()
    generator.manual_seed(config.training.seed)

    samples = torch.randn(32, 2, generator=generator)
    targets = torch.tensor([0, 1] * 16, dtype=torch.long)

    dataset = FedMedDataset(
        samples=samples,
        targets=targets,
        name="e9_centralized_train",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    torch.manual_seed(100)

    model = FlowerSmokeTestModel(
        name="e9_centralized",
        device="cpu",
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = SGD(
        model.parameters(),
        lr=config.training.learning_rate,
    )

    centralized_config = replace(
        config.training,
        local_epochs=config.federated.num_rounds * config.training.local_epochs,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        config=centralized_config,
    )

    evaluator = Evaluator(
        model=model,
        criterion=criterion,
        metrics=[Accuracy()],
    )

    training_result = trainer.train(dataloader)
    evaluation_result = evaluator.evaluate(dataloader)

    print("\n========== E9-A CENTRALIZED ==========")
    print(f"epochs: {training_result.epochs_completed}")
    print(f"samples_processed: {training_result.samples_processed}")
    print(f"batches_processed: {training_result.batches_processed}")
    print(f"final_train_loss: {training_result.final_loss}")
    print(f"eval_samples: {evaluation_result.samples_evaluated}")
    print(f"eval_batches: {evaluation_result.batches_evaluated}")
    print(f"eval_loss: {evaluation_result.loss}")
    print(f"eval_metrics: {evaluation_result.metrics}")
    print("======================================")

    assert training_result.epochs_completed == 15
    assert training_result.samples_processed == 32 * 15
    assert training_result.batches_processed == 8 * 15
    assert evaluation_result.samples_evaluated == 32
