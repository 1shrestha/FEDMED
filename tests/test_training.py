"""
Tests for the FedMed local training engine (src/training/trainer.py).

Uses a tiny synthetic classification model/dataset. No medical data.
"""

import pytest
import torch
from torch import nn

from src.common.config import TrainingConfig
from src.common.exceptions import TrainingError
from src.data.dataset import FedMedDataset
from src.data.loader import create_dataloader
from src.data.partitioner import partition_dataset
from src.models.base_model import BaseModel
from src.training.trainer import Trainer, TrainingResult, _infer_batch_size, _move_to_device


class TinyClassifier(BaseModel):
    """Minimal synthetic classification model used only for testing."""

    def build(self) -> nn.Module:
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def _make_synthetic_dataset(size: int = 20, name: str = "synthetic") -> FedMedDataset:
    torch.manual_seed(0)
    samples = [torch.randn(4) for _ in range(size)]
    targets = [i % 2 for i in range(size)]
    return FedMedDataset(samples=samples, targets=targets, name=name)


@pytest.fixture
def model() -> TinyClassifier:
    return TinyClassifier(name="tiny_classifier", device="cpu")


@pytest.fixture
def criterion():
    return nn.CrossEntropyLoss()


@pytest.fixture
def dataloader():
    dataset = _make_synthetic_dataset(size=20)
    return create_dataloader(dataset, batch_size=4, shuffle=False)


@pytest.fixture
def training_config() -> TrainingConfig:
    return TrainingConfig(
        local_epochs=2,
        batch_size=4,
        learning_rate=0.01,
        optimizer="adam",
        seed=42,
    )


# ============================================================
# Construction validation
# ============================================================


def test_valid_trainer_construction_with_config(model, criterion, training_config):
    """Verify Trainer can be constructed from model + criterion + config."""

    trainer = Trainer(model=model, criterion=criterion, config=training_config)

    assert isinstance(trainer, Trainer)


def test_valid_trainer_construction_with_explicit_optimizer(model, criterion):
    """Verify Trainer can be constructed with an explicit optimizer instead
    of a config."""

    optimizer = torch.optim.SGD(model.network.parameters(), lr=0.01)

    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer)

    assert isinstance(trainer, Trainer)


def test_trainer_requires_optimizer_or_config(model, criterion):
    """Verify Trainer construction fails without optimizer or config."""

    with pytest.raises(TrainingError):
        Trainer(model=model, criterion=criterion)


def test_invalid_model_rejected(criterion, training_config):
    """Verify a non-BaseModel object is rejected."""

    with pytest.raises(TrainingError):
        Trainer(model=object(), criterion=criterion, config=training_config)


def test_invalid_criterion_rejected(model, training_config):
    """Verify a non-callable criterion is rejected."""

    with pytest.raises(TrainingError):
        Trainer(model=model, criterion="not_callable", config=training_config)


def test_invalid_optimizer_rejected(model, criterion):
    """Verify a non-Optimizer object passed as optimizer is rejected."""

    with pytest.raises(TrainingError):
        Trainer(model=model, criterion=criterion, optimizer="not_an_optimizer")


def test_unsupported_optimizer_name_rejected(model, criterion):
    """Verify an unsupported optimizer name in config raises TrainingError."""

    bad_config = TrainingConfig(
        local_epochs=1,
        batch_size=4,
        learning_rate=0.01,
        optimizer="not_a_real_optimizer",
        seed=42,
    )

    with pytest.raises(TrainingError):
        Trainer(model=model, criterion=criterion, config=bad_config)


def test_invalid_config_type_rejected(model, criterion):
    """Verify a non-TrainingConfig object passed as config is rejected."""

    with pytest.raises(TrainingError):
        Trainer(model=model, criterion=criterion, config={"local_epochs": 1})


# ============================================================
# Optimizer/model compatibility (mandatory: Section 45)
# ============================================================


def test_optimizer_belonging_to_different_model_rejected(criterion):
    """
    An optimizer built for model_a must be rejected when supplied to
    a Trainer for model_b, and must be accepted for model_a itself.
    """

    model_a = TinyClassifier(name="model_a", device="cpu")
    model_b = TinyClassifier(name="model_b", device="cpu")

    optimizer_a = torch.optim.SGD(model_a.network.parameters(), lr=0.01)

    with pytest.raises(TrainingError):
        Trainer(model=model_b, criterion=criterion, optimizer=optimizer_a)

    trainer = Trainer(model=model_a, criterion=criterion, optimizer=optimizer_a)
    assert isinstance(trainer, Trainer)


def test_optimizer_with_only_unrelated_parameters_rejected(model, criterion):
    """An optimizer over parameters unrelated to the model must be rejected."""

    unrelated_parameter = torch.nn.Parameter(torch.randn(3, 3))
    optimizer = torch.optim.SGD([unrelated_parameter], lr=0.01)

    with pytest.raises(TrainingError):
        Trainer(model=model, criterion=criterion, optimizer=optimizer)


def test_optimizer_with_legitimately_frozen_parameters_allowed(criterion):
    """
    An optimizer covering only a model's trainable parameters (with
    some parameters legitimately frozen) must be accepted.
    """

    frozen_model = TinyClassifier(name="frozen_test_model", device="cpu")

    parameters = list(frozen_model.network.parameters())
    parameters[0].requires_grad = False  # freeze first layer's weight

    trainable_parameters = [p for p in frozen_model.network.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_parameters, lr=0.01)

    trainer = Trainer(model=frozen_model, criterion=criterion, optimizer=optimizer)
    assert isinstance(trainer, Trainer)


# ============================================================
# TrainingConfig field validation
# ============================================================


def test_zero_learning_rate_rejected(model, criterion):
    config = TrainingConfig(
        local_epochs=1, batch_size=4, learning_rate=0.0, optimizer="adam", seed=1
    )
    with pytest.raises(TrainingError):
        Trainer(model=model, criterion=criterion, config=config)


def test_negative_learning_rate_rejected(model, criterion):
    config = TrainingConfig(
        local_epochs=1, batch_size=4, learning_rate=-0.1, optimizer="adam", seed=1
    )
    with pytest.raises(TrainingError):
        Trainer(model=model, criterion=criterion, config=config)


def test_nan_learning_rate_rejected(model, criterion):
    config = TrainingConfig(
        local_epochs=1,
        batch_size=4,
        learning_rate=float("nan"),
        optimizer="adam",
        seed=1,
    )
    with pytest.raises(TrainingError):
        Trainer(model=model, criterion=criterion, config=config)


def test_infinite_learning_rate_rejected(model, criterion):
    config = TrainingConfig(
        local_epochs=1,
        batch_size=4,
        learning_rate=float("inf"),
        optimizer="adam",
        seed=1,
    )
    with pytest.raises(TrainingError):
        Trainer(model=model, criterion=criterion, config=config)


def test_zero_local_epochs_in_config_rejected(model, criterion):
    config = TrainingConfig(
        local_epochs=0, batch_size=4, learning_rate=0.01, optimizer="adam", seed=1
    )
    with pytest.raises(TrainingError):
        Trainer(model=model, criterion=criterion, config=config)


def test_negative_local_epochs_in_config_rejected(model, criterion):
    config = TrainingConfig(
        local_epochs=-2, batch_size=4, learning_rate=0.01, optimizer="adam", seed=1
    )
    with pytest.raises(TrainingError):
        Trainer(model=model, criterion=criterion, config=config)


def test_empty_optimizer_name_rejected(model, criterion):
    config = TrainingConfig(
        local_epochs=1, batch_size=4, learning_rate=0.01, optimizer="   ", seed=1
    )
    with pytest.raises(TrainingError):
        Trainer(model=model, criterion=criterion, config=config)


# ============================================================
# train() input validation
# ============================================================


def test_invalid_dataloader_rejected(model, criterion, training_config):
    """Verify a non-DataLoader object passed to train() is rejected."""

    trainer = Trainer(model=model, criterion=criterion, config=training_config)

    with pytest.raises(TrainingError):
        trainer.train(dataloader=[1, 2, 3], epochs=1)


def test_invalid_epoch_count_rejected(model, criterion, dataloader, training_config):
    """Verify zero/negative/non-int epoch counts are rejected."""

    trainer = Trainer(model=model, criterion=criterion, config=training_config)

    with pytest.raises(TrainingError):
        trainer.train(dataloader, epochs=0)

    with pytest.raises(TrainingError):
        trainer.train(dataloader, epochs=-1)

    with pytest.raises(TrainingError):
        trainer.train(dataloader, epochs="two")

    with pytest.raises(TrainingError):
        trainer.train(dataloader, epochs=True)  # bool rejected despite int subclass


def test_epochs_required_without_config(model, criterion, dataloader):
    """Verify epochs must be explicit when Trainer has no config."""

    optimizer = torch.optim.SGD(model.network.parameters(), lr=0.01)
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer)

    with pytest.raises(TrainingError):
        trainer.train(dataloader)  # no epochs, no config


# ============================================================
# Training behavior
# ============================================================


def test_valid_single_epoch_training(model, criterion, dataloader, training_config):
    """Verify a single epoch of training runs and returns a TrainingResult."""

    trainer = Trainer(model=model, criterion=criterion, config=training_config)

    result = trainer.train(dataloader, epochs=1)

    assert isinstance(result, TrainingResult)
    assert result.epochs_completed == 1


def test_multiple_epochs(model, criterion, dataloader, training_config):
    """Verify multiple epochs run and produce one loss entry per epoch."""

    trainer = Trainer(model=model, criterion=criterion, config=training_config)

    result = trainer.train(dataloader, epochs=3)

    assert result.epochs_completed == 3
    assert len(result.epoch_losses) == 3


def test_epochs_default_from_config(model, criterion, dataloader, training_config):
    """Verify epochs defaults to config.local_epochs when not passed explicitly."""

    trainer = Trainer(model=model, criterion=criterion, config=training_config)

    result = trainer.train(dataloader)  # no epochs arg; config.local_epochs == 2

    assert result.epochs_completed == training_config.local_epochs


def test_parameters_change_after_training(model, criterion, dataloader, training_config):
    """Verify model parameters actually update after training."""

    original_params = [p.clone() for p in model.network.parameters()]

    trainer = Trainer(model=model, criterion=criterion, config=training_config)
    trainer.train(dataloader, epochs=2)

    updated_params = list(model.network.parameters())

    assert any(
        not torch.equal(original, updated)
        for original, updated in zip(original_params, updated_params)
    )


def test_loss_is_produced(model, criterion, dataloader, training_config):
    """Verify training produces finite, non-negative loss values."""

    trainer = Trainer(model=model, criterion=criterion, config=training_config)

    result = trainer.train(dataloader, epochs=1)

    assert result.final_loss >= 0.0
    assert result.final_loss == result.epoch_losses[-1]
    assert all(loss >= 0.0 for loss in result.epoch_losses)


def test_samples_processed_is_correct(model, criterion, dataloader, training_config):
    """Verify samples_processed matches dataset size * epochs."""

    trainer = Trainer(model=model, criterion=criterion, config=training_config)

    result = trainer.train(dataloader, epochs=2)

    # dataset has 20 samples, 2 epochs -> 40 samples processed
    assert result.samples_processed == 40


def test_batches_processed_is_correct(model, criterion, dataloader, training_config):
    """Verify batches_processed matches expected batch count * epochs."""

    trainer = Trainer(model=model, criterion=criterion, config=training_config)

    result = trainer.train(dataloader, epochs=2)

    # dataset size 20, batch_size 4 -> 5 batches/epoch * 2 epochs = 10
    assert result.batches_processed == 10


def test_epoch_losses_length_is_correct(model, criterion, dataloader, training_config):
    """Verify epoch_losses has exactly one entry per completed epoch."""

    trainer = Trainer(model=model, criterion=criterion, config=training_config)

    result = trainer.train(dataloader, epochs=4)

    assert len(result.epoch_losses) == 4


def test_final_loss_matches_last_epoch(model, criterion, dataloader, training_config):
    """Verify final_loss is exactly the last entry of epoch_losses."""

    trainer = Trainer(model=model, criterion=criterion, config=training_config)

    result = trainer.train(dataloader, epochs=3)

    assert result.final_loss == result.epoch_losses[-1]


def test_model_enters_training_mode(model, criterion, dataloader, training_config):
    """Verify the model is in training mode during/after train()."""

    model.eval_mode()
    assert model.network.training is False

    trainer = Trainer(model=model, criterion=criterion, config=training_config)
    trainer.train(dataloader, epochs=1)

    assert model.network.training is True


def test_training_failure_becomes_training_error(model, dataloader, training_config):
    """Verify a criterion that raises mid-training surfaces as TrainingError."""

    def broken_criterion(outputs, targets):
        raise RuntimeError("simulated criterion failure")

    trainer = Trainer(model=model, criterion=broken_criterion, config=training_config)

    with pytest.raises(TrainingError):
        trainer.train(dataloader, epochs=1)


def test_dataloader_integration(model, criterion, training_config):
    """Verify Trainer works correctly with a directly-constructed DataLoader
    (not just the fixture)."""

    dataset = _make_synthetic_dataset(size=12)
    loader = create_dataloader(dataset, batch_size=3, shuffle=True)

    trainer = Trainer(model=model, criterion=criterion, config=training_config)
    result = trainer.train(loader, epochs=1)

    assert result.samples_processed == 12
    assert result.batches_processed == 4


# ============================================================
# Loss validation / gradient safety
# ============================================================


def test_non_scalar_loss_rejected(model, dataloader, training_config):
    """A criterion returning a non-scalar tensor must be rejected."""

    def bad_criterion(outputs, targets):
        return outputs.sum(dim=0)  # shape (num_classes,), not scalar

    trainer = Trainer(model=model, criterion=bad_criterion, config=training_config)

    with pytest.raises(TrainingError):
        trainer.train(dataloader, epochs=1)


def test_non_finite_loss_rejected(model, dataloader, training_config):
    """A criterion returning a NaN loss must be rejected, not silently continued."""

    def nan_criterion(outputs, targets):
        return outputs.sum() * float("nan")

    trainer = Trainer(model=model, criterion=nan_criterion, config=training_config)

    with pytest.raises(TrainingError):
        trainer.train(dataloader, epochs=1)


def test_backward_failure_becomes_chained_training_error(
    model, dataloader, training_config
):
    """A backward() failure must surface as TrainingError with the original
    exception chained via `__cause__`."""

    class _FailingBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.sum()

        @staticmethod
        def backward(ctx, grad_output):
            raise RuntimeError("simulated backward failure")

    def failing_criterion(outputs, targets):
        return _FailingBackward.apply(outputs)

    trainer = Trainer(model=model, criterion=failing_criterion, config=training_config)

    with pytest.raises(TrainingError) as excinfo:
        trainer.train(dataloader, epochs=1)

    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, RuntimeError)


# ============================================================
# Sample-weighted loss (mandatory: Section 44)
# ============================================================


def test_uneven_final_batch_produces_sample_weighted_loss(model, training_config):
    """
    Deterministic proof that epoch loss is sample-weighted, not a
    naive average of per-batch means. 5 samples, batch_size=3 ->
    batches of size 3 and 2. A fixed-value criterion makes the exact
    expected loss known; an unweighted-average implementation would
    fail this assertion.
    """

    dataset = _make_synthetic_dataset(size=5, name="uneven_loss_dataset")
    loader = create_dataloader(dataset, batch_size=3, shuffle=False, drop_last=False)

    loss_by_batch_size = {3: 0.2, 2: 1.0}

    def fixed_value_criterion(outputs, targets):
        batch_size = outputs.shape[0]
        # Tied to outputs' graph (requires_grad True, differentiable)
        # but numerically exactly loss_by_batch_size[batch_size].
        return outputs.sum() * 0.0 + torch.tensor(
            loss_by_batch_size[batch_size], dtype=outputs.dtype
        )

    trainer = Trainer(model=model, criterion=fixed_value_criterion, config=training_config)
    result = trainer.train(loader, epochs=1)

    expected_sample_weighted_loss = (0.2 * 3 + 1.0 * 2) / 5
    naive_unweighted_average = (0.2 + 1.0) / 2

    assert result.epoch_losses[0] == pytest.approx(expected_sample_weighted_loss)
    assert result.epoch_losses[0] != pytest.approx(naive_unweighted_average)


def test_samples_processed_correctness_with_uneven_batches(
    model, criterion, training_config
):
    """Verify samples_processed/batches_processed are correct for a dataset
    size not evenly divisible by batch_size."""

    dataset = _make_synthetic_dataset(size=7, name="uneven_batches_dataset")
    loader = create_dataloader(dataset, batch_size=3, shuffle=False, drop_last=False)

    trainer = Trainer(model=model, criterion=criterion, config=training_config)
    result = trainer.train(loader, epochs=1)

    assert result.samples_processed == 7
    assert result.batches_processed == 3  # batches of size 3, 3, 1


# ============================================================
# Empty DataLoader
# ============================================================


def test_batch_size_larger_than_dataset_with_drop_last_zero_batches(
    model, criterion, training_config
):
    """batch_size larger than the dataset with drop_last=True yields zero
    batches; training must reject this rather than reporting a fake epoch."""

    dataset = _make_synthetic_dataset(size=5, name="zero_batch_dataset")
    loader = create_dataloader(dataset, batch_size=10, drop_last=True)

    trainer = Trainer(model=model, criterion=criterion, config=training_config)

    with pytest.raises(TrainingError):
        trainer.train(loader, epochs=1)


# ============================================================
# Internal helpers (direct unit tests)
# ============================================================


def test_move_to_device_handles_nested_dict_list_tuple():
    """_move_to_device must recurse through dict/list/tuple, preserving
    structure and non-tensor leaves."""

    device = torch.device("cpu")

    batch = {
        "image": torch.randn(2, 3),
        "meta": [torch.randn(2), torch.randn(2)],
        "extra": (torch.randn(2), "non_tensor_value"),
    }

    moved = _move_to_device(batch, device)

    assert isinstance(moved, dict)
    assert set(moved.keys()) == {"image", "meta", "extra"}
    assert moved["image"].device == device

    assert isinstance(moved["meta"], list)
    assert all(t.device == device for t in moved["meta"])

    assert isinstance(moved["extra"], tuple)
    assert moved["extra"][0].device == device
    assert moved["extra"][1] == "non_tensor_value"


def test_infer_batch_size_rejects_scalar_tensor():
    """A 0-dimensional Tensor has no batch dimension and must raise
    TrainingError rather than crashing on shape[0]."""

    scalar_tensor = torch.tensor(5.0)

    with pytest.raises(TrainingError):
        _infer_batch_size(scalar_tensor, scalar_tensor)

def test_infer_batch_size_from_dict_shaped_samples():
    """
    A dict-shaped samples value, such as a collated multimodal batch,
    must infer batch size from its tensor values rather than len(dict).

    Example:
        {
            "image": Tensor[B, C, H, W],
            "meta": Tensor[B, D],
        }

    len(dict) would return the number of fields, not B.
    """

    dict_batch = {
        "image": torch.randn(6, 3, 8, 8),
        "meta": torch.randn(6, 4),
    }

    targets = torch.randint(0, 2, (6,))

    assert _infer_batch_size(dict_batch, targets) == 6


def test_infer_batch_size_from_nested_dict():
    """
    Batch-size inference must work when multimodal/nested sample
    structures contain mappings inside mappings.
    """

    nested_batch = {
        "image": {
            "left": torch.randn(5, 3, 8, 8),
            "right": torch.randn(5, 3, 8, 8),
        },
        "metadata": {
            "clinical": torch.randn(5, 4),
        },
    }

    targets = torch.randint(0, 2, (5,))

    assert _infer_batch_size(nested_batch, targets) == 5


def test_infer_batch_size_falls_back_from_dict_to_targets():
    """
    If a mapping contains no usable sized value, batch-size inference
    must fall back to targets rather than incorrectly using len(dict).
    """

    samples = {
        "metadata": "not_a_batch_dimension",
    }

    targets = torch.randint(0, 2, (7,))

    assert _infer_batch_size(samples, targets) == 7

def test_infer_batch_size_from_mapping():
    """
    Batch-size inference should work with mapping-compatible
    containers, not only the built-in dict type.
    """

    from collections import OrderedDict

    samples = OrderedDict(
        [
            ("image", torch.randn(4, 3, 8, 8)),
            ("meta", torch.randn(4, 2)),
        ]
    )

    targets = torch.randint(0, 2, (4,))

    assert _infer_batch_size(samples, targets) == 4    
# ============================================================
# End-to-end: PartitionView + DataLoader + Trainer
# ============================================================


def test_partition_view_dataloader_trainer_integration(model, criterion, training_config):
    """
    Verify the full chain: FedMedDataset -> partition_dataset() ->
    PartitionView -> create_dataloader() -> Trainer works end to end,
    not merely in isolated unit tests.
    """

    dataset = _make_synthetic_dataset(size=30, name="integration_dataset")
    partitions = partition_dataset(dataset, num_clients=3, seed=7)

    client_partition = partitions["client_0"]
    loader = create_dataloader(client_partition.dataset, batch_size=5, shuffle=True)

    trainer = Trainer(model=model, criterion=criterion, config=training_config)
    result = trainer.train(loader, epochs=1)

    assert result.samples_processed == len(client_partition.dataset)
    assert result.batches_processed == 2  # 10 samples / batch_size 5