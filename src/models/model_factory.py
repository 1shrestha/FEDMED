"""
Model factory for FedMed.

This module provides a centralized mechanism for creating concrete
FedMed model implementations.

The factory keeps model selection separate from the rest of the
application. Training, evaluation, and federated-learning layers
can request a model without knowing the concrete implementation.

The factory does NOT:

- train models
- load datasets
- perform federated aggregation
- contain Flower-specific logic
- define medical model architectures
- manage optimizers
"""

from __future__ import annotations

from src.common.exceptions import ModelError
from src.models.base_model import BaseModel


class ModelFactory:
    """
    Central registry and factory for FedMed models.

    Concrete BaseModel implementations are registered using a
    string identifier and can later be instantiated by name.
    """

    _registry: dict[str, type[BaseModel]] = {}

    @classmethod
    def register(
        cls,
        model_name: str,
        model_class: type[BaseModel],
    ) -> None:
        """
        Register a model implementation.

        Args:
            model_name:
                Unique identifier used to create the model.

            model_class:
                Concrete BaseModel subclass.

        Raises:
            ModelError:
                If the model name is invalid, already registered,
                or the supplied class does not inherit from BaseModel.
        """

        if not model_name or not model_name.strip():
            raise ModelError("Model name cannot be empty.")

        if not isinstance(model_class, type):
            raise ModelError(
                "Model class must be a class."
            )

        if not issubclass(model_class, BaseModel):
            raise ModelError(
                f"Model class '{model_class.__name__}' must "
                f"inherit from BaseModel."
            )

        normalized_name = model_name.strip().lower()

        if normalized_name in cls._registry:
            raise ModelError(
                f"Model '{normalized_name}' is already registered."
            )

        cls._registry[normalized_name] = model_class

    @classmethod
    def create(
        cls,
        model_name: str,
        **kwargs,
    ) -> BaseModel:
        """
        Create an instance of a registered model.

        Args:
            model_name:
                Identifier of the registered model.

            **kwargs:
                Arguments passed to the model constructor.

        Returns:
            A concrete BaseModel instance.

        Raises:
            ModelError:
                If the requested model is not registered or
                construction fails.
        """

        if not model_name or not model_name.strip():
            raise ModelError("Model name cannot be empty.")

        normalized_name = model_name.strip().lower()

        if normalized_name not in cls._registry:
            available_models = cls.available_models()

            available = (
                ", ".join(available_models)
                if available_models
                else "none"
            )

            raise ModelError(
                f"Unknown model '{model_name}'. "
                f"Available models: {available}"
            )

        model_class = cls._registry[normalized_name]

        try:
            return model_class(**kwargs)

        except ModelError:
            raise

        except Exception as exc:
            raise ModelError(
                f"Failed to create model '{normalized_name}': {exc}"
            ) from exc

    @classmethod
    def available_models(cls) -> list[str]:
        """
        Return all registered model identifiers.

        Returns:
            Sorted list of registered model names.
        """

        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(
        cls,
        model_name: str,
    ) -> bool:
        """
        Check whether a model is registered.

        Args:
            model_name:
                Model identifier.

        Returns:
            True if the model is registered, otherwise False.
        """

        if not model_name:
            return False

        return model_name.strip().lower() in cls._registry

    @classmethod
    def clear_registry(cls) -> None:
        """
        Clear the model registry.

        Primarily intended for isolated tests.

        This should not normally be called during application runtime.
        """

        cls._registry.clear()