import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path as FilePath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 32
    gradient_clip: Optional[float] = None
    device: str = "cpu"
    seed: int = 42
    early_stopping_patience: Optional[int] = None

    def validate(self):
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")

@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    duration_seconds: float = 0.0

@dataclass
class TrainingHistory:
    epochs: List[EpochResult] = field(default_factory=list)

    def add(self, result: EpochResult):
        self.epochs.append(result)

    def best_validation_accuracy(self):
        values = [
            x.validation_accuracy for x in self.epochs
            if x.validation_accuracy is not None
        ]
        return max(values) if values else None

    def best_validation_loss(self):
        values = [
            x.validation_loss for x in self.epochs
            if x.validation_loss is not None
        ]
        return min(values) if values else None

    def to_dict(self):
        return {"epochs": [asdict(x) for x in self.epochs]}

    def save_json(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

# ---------------------------------------------------------------------------
# Reproducibility and device helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_device(device: str = "cpu"):
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)

def move_batch(batch, device):
    if isinstance(batch, (tuple, list)):
        return tuple(
            item.to(device) if torch.is_tensor(item) else item
            for item in batch
        )
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch

# ---------------------------------------------------------------------------
# Model parameter utilities
# ---------------------------------------------------------------------------

def get_parameters(model: nn.Module):
    return [
        tensor.detach().cpu().clone()
        for tensor in model.state_dict().values()
    ]

def set_parameters(model: nn.Module, parameters):
    state = model.state_dict()
    if len(state) != len(parameters):
        raise ValueError("Parameter count does not match model state")
    updated = {
        key: value.detach().clone().to(state[key].device)
        for key, value in zip(state.keys(), parameters)
    }
    model.load_state_dict(updated, strict=True)

def clone_model(model: nn.Module):
    return copy.deepcopy(model)

def count_parameters(model: nn.Module, trainable_only=False):
    values = model.parameters()
    if trainable_only:
        values = (p for p in values if p.requires_grad)
    return sum(p.numel() for p in values)

def model_parameter_shapes(model: nn.Module):
    return {
        name: tuple(parameter.shape)
        for name, parameter in model.named_parameters()
    }

def model_parameter_norm(model: nn.Module):
    total = torch.tensor(0.0)
    for parameter in model.parameters():
        total += torch.sum(parameter.detach().float() ** 2)
    return float(torch.sqrt(total))

# ---------------------------------------------------------------------------
# Federated aggregation
# ---------------------------------------------------------------------------

def average_parameters(parameter_sets):
    if not parameter_sets:
        raise ValueError("parameter_sets cannot be empty")
    length = len(parameter_sets[0])
    if any(len(values) != length for values in parameter_sets):
        raise ValueError("All parameter sets must have equal length")
    result = []
    for values in zip(*parameter_sets):
        stacked = torch.stack([
            value.float().cpu() for value in values
        ])
        result.append(stacked.mean(dim=0))
    return result

def weighted_average_parameters(parameter_sets, sample_counts):
    if not parameter_sets:
        raise ValueError("parameter_sets cannot be empty")
    if len(parameter_sets) != len(sample_counts):
        raise ValueError("parameter sets and sample counts differ")
    if any(count < 0 for count in sample_counts):
        raise ValueError("sample counts cannot be negative")
    total = sum(sample_counts)
    if total <= 0:
        raise ValueError("total sample count must be positive")
    result = []
    for values in zip(*parameter_sets):
        accumulator = torch.zeros_like(values[0], dtype=torch.float32)
        for value, count in zip(values, sample_counts):
            accumulator += value.float().cpu() * (count / total)
        result.append(accumulator)
    return result

def parameter_difference(before, after):
    if len(before) != len(after):
        raise ValueError("parameter lists differ in length")
    return [
        new.float().cpu() - old.float().cpu()
        for old, new in zip(before, after)
    ]

def apply_parameter_update(parameters, updates):
    if len(parameters) != len(updates):
        raise ValueError("parameter lists differ in length")
    return [
        parameter.float().cpu() + update.float().cpu()
        for parameter, update in zip(parameters, updates)
    ]

def parameter_update_norm(updates):
    total = torch.tensor(0.0)
    for update in updates:
        total += torch.sum(update.float() ** 2)
    return float(torch.sqrt(total))

# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def accuracy_from_predictions(predictions, labels):
    predictions = torch.as_tensor(predictions)
    labels = torch.as_tensor(labels)
    if labels.numel() == 0:
        return 0.0
    return float((predictions == labels).float().mean().item() * 100.0)

def predictions_from_logits(logits):
    return torch.argmax(logits, dim=1)

def accuracy_from_logits(logits, labels):
    return accuracy_from_predictions(
        predictions_from_logits(logits), labels
    )

def confusion_matrix(predictions, labels, num_classes):
    predictions = torch.as_tensor(predictions).long().view(-1)
    labels = torch.as_tensor(labels).long().view(-1)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for actual, predicted in zip(labels, predictions):
        if 0 <= actual < num_classes and 0 <= predicted < num_classes:
            matrix[actual, predicted] += 1
    return matrix

def precision_recall_f1(predictions, labels, num_classes):
    matrix = confusion_matrix(predictions, labels, num_classes)
    precision = []
    recall = []
    f1 = []
    for index in range(num_classes):
        tp = matrix[index, index].item()
        fp = matrix[:, index].sum().item() - tp
        fn = matrix[index, :].sum().item() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        score = 2 * p * r / (p + r) if p + r else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(score)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def validate_dataset(dataset, minimum_samples=1):
    if dataset is None:
        raise ValueError("dataset cannot be None")
    size = len(dataset)
    if size < minimum_samples:
        raise ValueError(
            f"dataset contains {size} samples; "
            f"minimum is {minimum_samples}"
        )
    return size

def split_indices(length, validation_fraction=0.2, seed=42):
    if length < 2:
        raise ValueError("at least two samples are required")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(length, generator=generator).tolist()
    split = int(length * (1 - validation_fraction))
    split = max(1, min(split, length - 1))
    return permutation[:split], permutation[split:]

def split_dataset(dataset, validation_fraction=0.2, seed=42):
    train_indices, validation_indices = split_indices(
        len(dataset), validation_fraction, seed
    )
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)

def make_loader(dataset, batch_size=32, shuffle=True, num_workers=0):
    validate_dataset(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_batch(model, images, labels, optimizer, criterion, device="cpu",
                gradient_clip=None):
    model.train()
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    if gradient_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
    accuracy = accuracy_from_logits(outputs, labels)
    return float(loss.item()), accuracy

def train_one_epoch_pipeline(model, dataloader, optimizer, criterion,
                             device="cpu", gradient_clip=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip
            )
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (
            torch.argmax(outputs, dim=1) == labels
        ).sum().item()
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("dataloader produced no samples")
    return (
        total_loss / total_samples,
        100.0 * total_correct / total_samples,
    )

@torch.no_grad()
def evaluate_pipeline(model, dataloader, criterion, device="cpu"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        predictions = torch.argmax(outputs, dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())
    if total_samples == 0:
        raise ValueError("dataloader produced no samples")
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    return {
        "loss": total_loss / total_samples,
        "accuracy": 100.0 * total_correct / total_samples,
        "predictions": predictions,
        "labels": labels,
    }

# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer=None, scheduler=None,
                    epoch=0, history=None, extra=None):
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "history": history.to_dict() if history else None,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None,
                    map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint

# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------

class ExperimentLogger:
    def __init__(self, path=None):
        self.path = path
        self.records = []

    def log(self, **values):
        record = dict(values)
        record["timestamp"] = time.time()
        self.records.append(record)
        if self.path:
            self.flush()

    def flush(self):
        if not self.path:
            return
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.records, handle, indent=2)

    def latest(self):
        return self.records[-1] if self.records else None

# ---------------------------------------------------------------------------
# Early stopping and schedulers
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=3, mode="min", min_delta=0.0):
        if patience < 1:
            raise ValueError("patience must be positive")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be min or max")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def update(self, value):
        if self.best is None:
            self.best = value
            return False
        if self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

def build_optimizer(model, learning_rate=0.001, weight_decay=0.0):
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

def build_scheduler(optimizer, step_size=5, gamma=0.5):
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )

def current_learning_rate(optimizer):
    return float(optimizer.param_groups[0]["lr"])

# ---------------------------------------------------------------------------
# Full training runner
# ---------------------------------------------------------------------------

def fit_model(model, trainloader, validation_loader=None, config=None):
    config = config or TrainingConfig()
    config.validate()
    set_seed(config.seed)
    device = resolve_device(config.device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        config.learning_rate,
        config.weight_decay,
    )

    history = TrainingHistory()
    stopper = (
        EarlyStopping(config.early_stopping_patience, mode="min")
        if config.early_stopping_patience
        else None
    )

    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch_pipeline(
            model,
            trainloader,
            optimizer,
            criterion,
            device,
            config.gradient_clip,
        )

        validation_loss = None
        validation_accuracy = None

        if validation_loader is not None:
            result = evaluate_pipeline(
                model,
                validation_loader,
                criterion,
                device,
            )
            validation_loss = result["loss"]
            validation_accuracy = result["accuracy"]

        duration = time.perf_counter() - start

        history.add(EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
            duration_seconds=duration,
        ))

        if stopper and validation_loss is not None:
            if stopper.update(validation_loss):
                break

    return history

def safe_mean(*args, **kwargs):
    """Small reusable FEDMED pipeline helper."""
    return float(np.mean(values)) if len(values) else 0.0

def safe_std(*args, **kwargs):
    """Small reusable FEDMED pipeline helper."""
    return float(np.std(values)) if len(values) else 0.0

def safe_min(*args, **kwargs):
    """Small reusable FEDMED pipeline helper."""
    return float(np.min(values)) if len(values) else 0.0

def safe_max(*args, **kwargs):
    """Small reusable FEDMED pipeline helper."""
    return float(np.max(values)) if len(values) else 0.0

def parameter_count(*args, **kwargs):
    """Small reusable FEDMED pipeline helper."""
    return count_parameters(model, False)

def trainable_parameter_count(*args, **kwargs):
    """Small reusable FEDMED pipeline helper."""
    return count_parameters(model, True)

def learning_rate(*args, **kwargs):
    """Small reusable FEDMED pipeline helper."""
    return current_learning_rate(optimizer)

def is_finite_loss(*args, **kwargs):
    """Small reusable FEDMED pipeline helper."""
    return math.isfinite(float(loss))

def dataset_size(*args, **kwargs):
    """Small reusable FEDMED pipeline helper."""
    return validate_dataset(dataset)

def parameter_list_size(*args, **kwargs):
    """Small reusable FEDMED pipeline helper."""
    return len(get_parameters(model))

def validate_pipeline_value_001(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 001.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 001 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 001 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 001 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 001 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 001 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 001 cannot be None")
    return True

def validate_pipeline_value_002(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 002.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 002 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 002 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 002 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 002 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 002 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 002 cannot be None")
    return True

def validate_pipeline_value_003(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 003.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 003 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 003 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 003 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 003 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 003 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 003 cannot be None")
    return True

def validate_pipeline_value_004(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 004.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 004 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 004 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 004 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 004 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 004 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 004 cannot be None")
    return True

def validate_pipeline_value_005(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 005.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 005 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 005 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 005 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 005 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 005 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 005 cannot be None")
    return True

def validate_pipeline_value_006(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 006.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 006 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 006 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 006 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 006 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 006 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 006 cannot be None")
    return True

def validate_pipeline_value_007(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 007.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 007 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 007 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 007 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 007 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 007 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 007 cannot be None")
    return True

def validate_pipeline_value_008(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 008.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 008 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 008 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 008 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 008 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 008 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 008 cannot be None")
    return True

def validate_pipeline_value_009(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 009.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 009 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 009 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 009 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 009 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 009 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 009 cannot be None")
    return True

def validate_pipeline_value_010(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 010.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 010 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 010 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 010 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 010 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 010 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 010 cannot be None")
    return True

def validate_pipeline_value_011(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 011.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 011 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 011 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 011 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 011 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 011 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 011 cannot be None")
    return True

def validate_pipeline_value_012(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 012.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 012 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 012 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 012 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 012 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 012 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 012 cannot be None")
    return True

def validate_pipeline_value_013(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 013.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 013 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 013 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 013 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 013 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 013 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 013 cannot be None")
    return True

def validate_pipeline_value_014(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 014.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 014 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 014 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 014 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 014 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 014 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 014 cannot be None")
    return True

def validate_pipeline_value_015(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 015.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 015 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 015 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 015 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 015 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 015 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 015 cannot be None")
    return True

def validate_pipeline_value_016(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 016.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 016 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 016 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 016 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 016 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 016 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 016 cannot be None")
    return True

def validate_pipeline_value_017(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 017.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 017 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 017 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 017 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 017 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 017 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 017 cannot be None")
    return True

def validate_pipeline_value_018(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 018.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 018 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 018 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 018 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 018 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 018 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 018 cannot be None")
    return True

def validate_pipeline_value_019(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 019.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 019 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 019 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 019 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 019 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 019 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 019 cannot be None")
    return True

def validate_pipeline_value_020(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 020.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 020 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 020 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 020 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 020 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 020 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 020 cannot be None")
    return True

def validate_pipeline_value_021(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 021.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 021 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 021 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 021 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 021 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 021 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 021 cannot be None")
    return True

def validate_pipeline_value_022(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 022.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 022 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 022 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 022 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 022 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 022 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 022 cannot be None")
    return True

def validate_pipeline_value_023(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 023.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 023 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 023 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 023 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 023 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 023 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 023 cannot be None")
    return True

def validate_pipeline_value_024(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 024.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 024 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 024 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 024 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 024 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 024 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 024 cannot be None")
    return True

def validate_pipeline_value_025(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 025.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 025 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 025 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 025 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 025 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 025 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 025 cannot be None")
    return True

def validate_pipeline_value_026(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 026.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 026 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 026 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 026 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 026 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 026 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 026 cannot be None")
    return True

def validate_pipeline_value_027(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 027.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 027 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 027 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 027 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 027 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 027 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 027 cannot be None")
    return True

def validate_pipeline_value_028(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 028.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 028 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 028 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 028 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 028 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 028 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 028 cannot be None")
    return True

def validate_pipeline_value_029(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 029.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 029 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 029 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 029 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 029 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 029 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 029 cannot be None")
    return True

def validate_pipeline_value_030(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 030.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 030 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 030 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 030 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 030 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 030 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 030 cannot be None")
    return True

def validate_pipeline_value_031(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 031.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 031 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 031 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 031 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 031 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 031 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 031 cannot be None")
    return True

def validate_pipeline_value_032(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 032.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 032 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 032 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 032 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 032 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 032 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 032 cannot be None")
    return True

def validate_pipeline_value_033(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 033.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 033 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 033 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 033 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 033 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 033 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 033 cannot be None")
    return True

def validate_pipeline_value_034(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 034.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 034 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 034 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 034 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 034 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 034 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 034 cannot be None")
    return True

def validate_pipeline_value_035(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 035.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 035 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 035 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 035 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 035 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 035 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 035 cannot be None")
    return True

def validate_pipeline_value_036(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 036.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 036 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 036 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 036 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 036 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 036 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 036 cannot be None")
    return True

def validate_pipeline_value_037(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 037.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 037 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 037 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 037 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 037 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 037 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 037 cannot be None")
    return True

def validate_pipeline_value_038(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 038.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 038 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 038 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 038 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 038 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 038 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 038 cannot be None")
    return True

def validate_pipeline_value_039(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 039.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 039 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 039 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 039 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 039 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 039 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 039 cannot be None")
    return True

def validate_pipeline_value_040(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 040.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 040 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 040 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 040 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 040 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 040 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 040 cannot be None")
    return True

def validate_pipeline_value_041(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 041.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 041 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 041 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 041 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 041 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 041 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 041 cannot be None")
    return True

def validate_pipeline_value_042(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 042.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 042 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 042 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 042 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 042 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 042 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 042 cannot be None")
    return True

def validate_pipeline_value_043(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 043.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 043 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 043 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 043 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 043 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 043 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 043 cannot be None")
    return True

def validate_pipeline_value_044(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 044.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 044 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 044 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 044 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 044 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 044 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 044 cannot be None")
    return True

def validate_pipeline_value_045(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 045.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 045 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 045 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 045 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 045 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 045 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 045 cannot be None")
    return True

def validate_pipeline_value_046(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 046.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 046 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 046 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 046 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 046 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 046 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 046 cannot be None")
    return True

def validate_pipeline_value_047(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 047.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 047 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 047 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 047 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 047 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 047 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 047 cannot be None")
    return True

def validate_pipeline_value_048(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 048.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 048 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 048 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 048 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 048 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 048 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 048 cannot be None")
    return True

def validate_pipeline_value_049(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 049.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 049 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 049 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 049 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 049 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 049 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 049 cannot be None")
    return True

def validate_pipeline_value_050(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 050.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 050 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 050 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 050 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 050 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 050 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 050 cannot be None")
    return True

def validate_pipeline_value_051(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 051.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 051 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 051 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 051 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 051 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 051 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 051 cannot be None")
    return True

def validate_pipeline_value_052(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 052.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 052 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 052 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 052 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 052 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 052 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 052 cannot be None")
    return True

def validate_pipeline_value_053(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 053.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 053 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 053 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 053 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 053 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 053 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 053 cannot be None")
    return True

def validate_pipeline_value_054(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 054.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 054 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 054 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 054 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 054 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 054 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 054 cannot be None")
    return True

def validate_pipeline_value_055(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 055.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 055 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 055 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 055 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 055 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 055 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 055 cannot be None")
    return True

def validate_pipeline_value_056(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 056.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 056 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 056 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 056 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 056 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 056 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 056 cannot be None")
    return True

def validate_pipeline_value_057(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 057.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 057 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 057 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 057 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 057 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 057 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 057 cannot be None")
    return True

def validate_pipeline_value_058(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 058.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 058 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 058 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 058 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 058 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 058 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 058 cannot be None")
    return True

def validate_pipeline_value_059(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 059.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 059 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 059 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 059 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 059 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 059 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 059 cannot be None")
    return True

def validate_pipeline_value_060(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 060.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 060 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 060 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 060 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 060 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 060 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 060 cannot be None")
    return True

def validate_pipeline_value_061(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 061.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 061 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 061 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 061 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 061 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 061 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 061 cannot be None")
    return True

def validate_pipeline_value_062(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 062.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 062 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 062 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 062 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 062 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 062 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 062 cannot be None")
    return True

def validate_pipeline_value_063(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 063.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 063 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 063 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 063 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 063 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 063 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 063 cannot be None")
    return True

def validate_pipeline_value_064(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 064.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 064 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 064 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 064 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 064 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 064 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 064 cannot be None")
    return True

def validate_pipeline_value_065(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 065.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 065 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 065 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 065 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 065 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 065 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 065 cannot be None")
    return True

def validate_pipeline_value_066(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 066.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 066 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 066 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 066 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 066 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 066 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 066 cannot be None")
    return True

def validate_pipeline_value_067(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 067.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 067 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 067 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 067 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 067 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 067 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 067 cannot be None")
    return True

def validate_pipeline_value_068(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 068.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 068 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 068 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 068 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 068 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 068 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 068 cannot be None")
    return True

def validate_pipeline_value_069(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 069.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 069 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 069 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 069 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 069 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 069 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 069 cannot be None")
    return True

def validate_pipeline_value_070(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 070.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 070 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 070 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 070 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 070 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 070 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 070 cannot be None")
    return True

def validate_pipeline_value_071(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 071.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 071 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 071 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 071 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 071 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 071 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 071 cannot be None")
    return True

def validate_pipeline_value_072(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 072.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 072 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 072 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 072 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 072 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 072 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 072 cannot be None")
    return True

def validate_pipeline_value_073(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 073.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 073 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 073 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 073 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 073 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 073 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 073 cannot be None")
    return True

def validate_pipeline_value_074(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 074.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 074 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 074 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 074 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 074 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 074 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 074 cannot be None")
    return True

def validate_pipeline_value_075(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 075.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 075 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 075 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 075 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 075 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 075 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 075 cannot be None")
    return True

def validate_pipeline_value_076(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 076.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 076 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 076 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 076 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 076 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 076 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 076 cannot be None")
    return True

def validate_pipeline_value_077(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 077.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 077 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 077 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 077 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 077 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 077 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 077 cannot be None")
    return True

def validate_pipeline_value_078(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 078.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 078 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 078 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 078 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 078 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 078 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 078 cannot be None")
    return True

def validate_pipeline_value_079(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 079.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 079 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 079 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 079 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 079 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 079 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 079 cannot be None")
    return True

def validate_pipeline_value_080(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 080.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 080 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 080 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 080 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 080 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 080 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 080 cannot be None")
    return True

def validate_pipeline_value_081(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 081.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 081 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 081 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 081 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 081 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 081 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 081 cannot be None")
    return True

def validate_pipeline_value_082(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 082.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 082 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 082 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 082 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 082 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 082 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 082 cannot be None")
    return True

def validate_pipeline_value_083(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 083.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 083 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 083 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 083 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 083 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 083 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 083 cannot be None")
    return True

def validate_pipeline_value_084(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 084.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 084 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 084 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 084 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 084 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 084 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 084 cannot be None")
    return True

def validate_pipeline_value_085(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 085.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 085 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 085 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 085 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 085 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 085 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 085 cannot be None")
    return True

def validate_pipeline_value_086(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 086.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 086 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 086 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 086 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 086 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 086 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 086 cannot be None")
    return True

def validate_pipeline_value_087(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 087.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 087 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 087 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 087 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 087 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 087 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 087 cannot be None")
    return True

def validate_pipeline_value_088(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 088.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 088 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 088 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 088 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 088 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 088 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 088 cannot be None")
    return True

def validate_pipeline_value_089(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 089.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 089 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 089 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 089 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 089 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 089 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 089 cannot be None")
    return True

def validate_pipeline_value_090(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 090.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 090 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 090 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 090 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 090 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 090 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 090 cannot be None")
    return True

def validate_pipeline_value_091(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 091.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 091 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 091 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 091 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 091 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 091 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 091 cannot be None")
    return True

def validate_pipeline_value_092(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 092.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 092 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 092 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 092 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 092 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 092 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 092 cannot be None")
    return True

def validate_pipeline_value_093(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 093.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 093 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 093 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 093 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 093 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 093 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 093 cannot be None")
    return True

def validate_pipeline_value_094(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 094.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 094 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 094 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 094 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 094 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 094 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 094 cannot be None")
    return True

def validate_pipeline_value_095(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 095.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 095 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 095 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 095 is empty")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("pipeline value 095 contains non-finite values")
        return True
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("pipeline value 095 is not finite")
        return True
    if value is None:
        raise ValueError("pipeline value 095 cannot be None")
    return True

def validate_pipeline_value_096(value):
    """
    Validate a pipeline scalar/collection for finite numeric content.
    Diagnostic helper 096.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("pipeline value 096 is empty")
        if not torch.isfinite(value.float()).all():
            raise ValueError("pipeline value 096 contains non-finite values")
        return True
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("pipeline value 096 is empty")