"""
Framework-independent federated client for FedMed.

Phase 3.2 defines the client-side orchestration boundary between
FedMed's existing local machine-learning foundation and the future
federated runtime.

The FederatedClient is responsible for:

- representing one federated participant
- receiving global model parameters
- validating and loading parameters through Phase 3.1
- invoking the existing Phase 2 Trainer
- invoking the existing Phase 2 Evaluator
- returning locally trained model parameters
- returning local sample counts
- returning local training/evaluation metrics

The FederatedClient intentionally does NOT:

- implement a training loop
- implement an evaluation loop
- create datasets
- partition datasets
- create DataLoaders
- perform aggregation
- implement FedAvg
- select clients
- manage federation rounds
- implement server logic
- implement networking
- import Flower-specific APIs

Flower-specific integration belongs at the outer federated-runtime
boundary. This module therefore remains framework-independent.

Architecture:

    Global Parameters
           |
           v
    FederatedClient
           |
      +----+----+
      |         |
      v         v
   Trainer   Evaluator
      |         |
      v         v
 TrainingResult
 EvaluationResult
      |         |
      +----+----+
           |
           v
    Federated Results
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from torch.utils.data import DataLoader

from src.common.exceptions import FederatedLearningError
from src.fl.parameters import (
    ParameterContract,
    ParameterPayload,
    copy_parameters,
    extract_parameters,
    load_parameters,
    validate_parameters,
)
from src.models.base_model import BaseModel
from src.training.evaluator import EvaluationResult, Evaluator
from src.training.trainer import Trainer, TrainingResult


# ============================================================
# Result types
# ============================================================


@dataclass(frozen=True)
class FederatedFitResult:
    """
    Result returned after one client's local training operation.

    Attributes:
        parameters:
            Updated model parameters after local training.

        num_examples:
            Number of local samples processed during training.

        metrics:
            Numeric local training metrics.

        epochs_completed:
            Number of local epochs completed.

        batches_processed:
            Number of batches processed during training.

        final_loss:
            Final epoch mean training loss.
    """

    parameters: ParameterPayload
    num_examples: int
    metrics: Mapping[str, float]
    epochs_completed: int
    batches_processed: int
    final_loss: float


@dataclass(frozen=True)
class FederatedEvaluateResult:
    """
    Result returned after one client's local evaluation operation.

    Attributes:
        num_examples:
            Number of samples evaluated.

        loss:
            Sample-weighted evaluation loss.

        metrics:
            Evaluation metrics produced by the existing Metric
            abstraction.
    """

    num_examples: int
    loss: float
    metrics: Mapping[str, float]


# ============================================================
# Federated Client
# ============================================================


class FederatedClient:
    """
    Framework-independent FedMed federated client.

    A FederatedClient coordinates one local model, its training
    DataLoader, optional evaluation DataLoader, Trainer, and
    Evaluator.

    The client represents one data-owning federated participant.
    Local data remains behind the DataLoader boundary.

    The same BaseModel instance must be used by:

        FederatedClient
            |
            +-- Trainer
            |
            +-- Evaluator

    This prevents a dangerous configuration where federated
    parameters are loaded into one model while Trainer or
    Evaluator operates on another model instance.
    """

    def __init__(
        self,
        client_id: str,
        model: BaseModel,
        trainer: Trainer,
        evaluator: Evaluator,
        train_loader: DataLoader,
        eval_loader: DataLoader | None = None,
    ) -> None:
        """
        Construct a validated federated client.

        Args:
            client_id:
                Stable identifier for the federated participant.

            model:
                BaseModel used by the client.

            trainer:
                Existing Phase 2 Trainer operating on the same model.

            evaluator:
                Existing Phase 2 Evaluator operating on the same model.

            train_loader:
                Local training DataLoader.

            eval_loader:
                Optional local evaluation DataLoader.

        Raises:
            FederatedLearningError:
                If a dependency is invalid, a local dataset is empty,
                or Trainer/Evaluator use a different model instance.
        """

        self._validate_client_id(client_id)

        if not isinstance(model, BaseModel):
            raise FederatedLearningError(
                "FederatedClient requires a BaseModel instance, "
                f"got {type(model).__name__}."
            )

        if not isinstance(trainer, Trainer):
            raise FederatedLearningError(
                "FederatedClient requires a Trainer instance, "
                f"got {type(trainer).__name__}."
            )

        if not isinstance(evaluator, Evaluator):
            raise FederatedLearningError(
                "FederatedClient requires an Evaluator instance, "
                f"got {type(evaluator).__name__}."
            )

        if not isinstance(train_loader, DataLoader):
            raise FederatedLearningError(
                "FederatedClient requires a training DataLoader, "
                f"got {type(train_loader).__name__}."
            )

        if eval_loader is not None and not isinstance(
            eval_loader,
            DataLoader,
        ):
            raise FederatedLearningError(
                "FederatedClient requires eval_loader to be a "
                f"DataLoader or None, got {type(eval_loader).__name__}."
            )

        # Trainer and Evaluator already validate their own model
        # dependencies during construction. Their model reference is
        # intentionally checked here because the client requires all
        # three components to operate on the exact same model object.
        if trainer._model is not model:
            raise FederatedLearningError(
                "FederatedClient model mismatch: Trainer is bound "
                "to a different model instance."
            )

        if evaluator._model is not model:
            raise FederatedLearningError(
                "FederatedClient model mismatch: Evaluator is bound "
                "to a different model instance."
            )

        if len(train_loader.dataset) == 0:
            raise FederatedLearningError(
                "FederatedClient requires a non-empty training dataset."
            )

        if (
            eval_loader is not None
            and len(eval_loader.dataset) == 0
        ):
            raise FederatedLearningError(
                "FederatedClient requires a non-empty evaluation "
                "dataset when eval_loader is provided."
            )

        self._client_id = client_id
        self._model = model
        self._trainer = trainer
        self._evaluator = evaluator
        self._train_loader = train_loader
        self._eval_loader = eval_loader

        # The model architecture and state layout are expected to
        # remain stable for the lifetime of this client.
        self._parameter_contract = ParameterContract.from_model(
            model
        )

    # ========================================================
    # Public properties
    # ========================================================

    @property
    def client_id(self) -> str:
        """Return the stable federated client identifier."""

        return self._client_id

    @property
    def parameter_contract(self) -> ParameterContract:
        """
        Return the immutable parameter contract for this client.
        """

        return self._parameter_contract

    @property
    def has_evaluator(self) -> bool:
        """
        Return whether a local evaluation DataLoader is configured.
        """

        return self._eval_loader is not None

    # ========================================================
    # Parameter operations
    # ========================================================

    def get_parameters(self) -> ParameterPayload:
        """
        Return the client's current model parameters.

        Extraction is performed through the Phase 3.1 parameter
        contract. A defensive copy is returned so callers cannot
        mutate model state through the returned arrays.

        Returns:
            Current model state as a list of NumPy arrays.

        Raises:
            FederatedLearningError:
                If the model state no longer matches the client's
                parameter contract.
        """

        try:
            parameters = extract_parameters(self._model)

            validate_parameters(
                parameters,
                self._parameter_contract,
            )

            return copy_parameters(parameters)

        except Exception as exc:
            if isinstance(exc, FederatedLearningError):
                raise

            raise FederatedLearningError(
                f"Client '{self._client_id}' failed to extract "
                f"model parameters: {exc}"
            ) from exc

    def set_parameters(
        self,
        parameters: Sequence[np.ndarray],
    ) -> None:
        """
        Validate and load global federated parameters.

        Parameters:
            parameters:
                Incoming global model parameter payload.

        Raises:
            FederatedLearningError:
                If the payload violates the client's parameter
                contract or cannot be loaded.
        """

        self._validate_parameters(parameters)

        try:
            load_parameters(
                self._model,
                parameters,
            )
        except Exception as exc:
            if isinstance(exc, FederatedLearningError):
                raise

            raise FederatedLearningError(
                f"Client '{self._client_id}' failed to load "
                f"federated parameters: {exc}"
            ) from exc

    # ========================================================
    # Local training
    # ========================================================

    def fit(
        self,
        parameters: Sequence[np.ndarray],
    ) -> FederatedFitResult:
        """
        Perform one local federated training operation.

        Lifecycle:

            incoming global parameters
                    |
                    v
              validation/load
                    |
                    v
              existing Trainer
                    |
                    v
              local TrainingResult
                    |
                    v
              updated parameters

        Parameters:
            parameters:
                Global model parameters supplied by the federated
                runtime.

        Returns:
            FederatedFitResult containing:

            - updated parameters
            - number of local examples
            - training metrics
            - completed epochs
            - processed batches
            - final loss

        Raises:
            FederatedLearningError:
                If parameters are invalid or local training fails.
        """

        self.set_parameters(parameters)

        try:
            result = self._trainer.train(
                self._train_loader,
            )
        except Exception as exc:
            raise FederatedLearningError(
                f"Federated local training failed for client "
                f"'{self._client_id}': {exc}"
            ) from exc

        if not isinstance(result, TrainingResult):
            raise FederatedLearningError(
                "Trainer.train() returned an unexpected result type: "
                f"{type(result).__name__}."
            )

        updated_parameters = self.get_parameters()

        metrics = self._build_training_metrics(result)

        return FederatedFitResult(
            parameters=updated_parameters,
            num_examples=int(result.samples_processed),
            metrics=metrics,
            epochs_completed=int(result.epochs_completed),
            batches_processed=int(result.batches_processed),
            final_loss=float(result.final_loss),
        )

    # ========================================================
    # Local evaluation
    # ========================================================

    def evaluate(
        self,
        parameters: Sequence[np.ndarray],
    ) -> FederatedEvaluateResult:
        """
        Evaluate supplied global parameters on local data.

        Evaluation is deliberately separate from fit(). This allows
        the future federated runtime/strategy to control when local
        evaluation occurs.

        Parameters:
            parameters:
                Model parameters to evaluate.

        Returns:
            FederatedEvaluateResult containing:

            - number of evaluated samples
            - evaluation loss
            - task metrics

        Raises:
            FederatedLearningError:
                If no evaluation loader is configured, parameters
                are invalid, or evaluation fails.
        """

        if self._eval_loader is None:
            raise FederatedLearningError(
                f"Client '{self._client_id}' has no evaluation "
                "DataLoader configured."
            )

        self.set_parameters(parameters)

        try:
            result = self._evaluator.evaluate(
                self._eval_loader,
            )
        except Exception as exc:
            raise FederatedLearningError(
                f"Federated local evaluation failed for client "
                f"'{self._client_id}': {exc}"
            ) from exc

        if not isinstance(result, EvaluationResult):
            raise FederatedLearningError(
                "Evaluator.evaluate() returned an unexpected "
                f"result type: {type(result).__name__}."
            )

        return FederatedEvaluateResult(
            num_examples=int(result.samples_evaluated),
            loss=float(result.loss),
            metrics=self._build_evaluation_metrics(result),
        )

    # ========================================================
    # Internal validation
    # ========================================================

    @staticmethod
    def _validate_client_id(client_id: str) -> None:
        """Validate the federated client identifier."""

        if not isinstance(client_id, str):
            raise FederatedLearningError(
                "client_id must be a string, "
                f"got {type(client_id).__name__}."
            )

        if not client_id.strip():
            raise FederatedLearningError(
                "client_id must be a non-empty string."
            )

    def _validate_parameters(
        self,
        parameters: Sequence[np.ndarray],
    ) -> None:
        """
        Validate parameters against this client's fixed model contract.

        All detailed parameter validation remains centralized in
        Phase 3.1's parameter module.
        """

        try:
            validate_parameters(
                parameters,
                self._parameter_contract,
            )
        except Exception as exc:
            if isinstance(exc, FederatedLearningError):
                raise

            raise FederatedLearningError(
                f"Parameter validation failed for client "
                f"'{self._client_id}': {exc}"
            ) from exc

    # ========================================================
    # Metric/result conversion
    # ========================================================

    @staticmethod
    def _build_training_metrics(
        result: TrainingResult,
    ) -> dict[str, float]:
        """
        Convert TrainingResult information into numeric metrics.

        Sample count, epoch count, and batch count remain explicit
        FederatedFitResult fields because later aggregation and
        federation monitoring give them semantic meaning.
        """

        return {
            "train_loss": float(result.final_loss),
            "epochs_completed": float(result.epochs_completed),
            "batches_processed": float(result.batches_processed),
        }

    @staticmethod
    def _build_evaluation_metrics(
        result: EvaluationResult,
    ) -> dict[str, float]:
        """
        Return a fresh snapshot of evaluator metrics.

        Evaluation loss remains a dedicated result field.
        Task-specific metrics remain inside the metrics mapping.
        """

        return {
            str(name): float(value)
            for name, value in result.metrics.items()
        }