import pytest
import torch

from src.common.exceptions import FederatedLearningError
from src.fl.orchestrator import FedMedOrchestrator


def test_orchestrator_can_be_constructed() -> None:
    orchestrator = FedMedOrchestrator()

    assert orchestrator is not None


def test_build_strategy_composes_fedavg_strategy_and_aggregator() -> None:
    from src.aggregation.fedavg import FedAvgAggregator
    from src.fl.strategy import FedAvgStrategy

    orchestrator = FedMedOrchestrator()

    strategy = orchestrator.build_strategy()

    assert isinstance(strategy, FedAvgStrategy)
    assert isinstance(strategy.aggregator, FedAvgAggregator)


def test_orchestrator_can_be_constructed_multiple_times() -> None:
    first = FedMedOrchestrator()
    second = FedMedOrchestrator()

    assert first is not second


def test_orchestrator_uses_single_torch_thread() -> None:
    orchestrator = FedMedOrchestrator()

    assert orchestrator is not None
    assert torch.get_num_threads() == 1


def test_partitioned_loader_uses_distinct_train_and_eval_splits() -> None:
    orchestrator = FedMedOrchestrator()

    train_loader = orchestrator._create_partitioned_loader(
        0,
        split="train",
    )
    eval_loader = orchestrator._create_partitioned_loader(
        0,
        split="eval",
    )

    assert len(train_loader.dataset) == 4
    assert len(eval_loader.dataset) == 8


def test_build_client_uses_eval_split_for_evaluation_loader() -> None:
    orchestrator = FedMedOrchestrator()

    client = orchestrator.build_client(
        "client_0",
        partition_index=0,
    )

    assert len(client._train_loader.dataset) == 4
    assert len(client._eval_loader.dataset) == 8


def test_build_client_uses_configured_optimizer() -> None:
    from src.common.config import TrainingConfig

    orchestrator = FedMedOrchestrator()

    orchestrator._training_config = TrainingConfig(
        local_epochs=2,
        batch_size=4,
        learning_rate=0.01,
        optimizer="adam",
        seed=42,
    )

    client = orchestrator.build_client(
        "client_0",
        partition_index=0,
    )

    assert isinstance(client._trainer._optimizer, torch.optim.Adam)


def test_build_client_uses_configured_model_settings() -> None:
    from src.common.config import ModelConfig

    orchestrator = FedMedOrchestrator()

    orchestrator._model_config = ModelConfig(
        name="configured_model",
        device="cpu:0",
    )

    client = orchestrator.build_client(
        "client_0",
        partition_index=0,
    )

    assert client._model.name == "configured_model_client_0"
    assert client._model.device == torch.device("cpu:0")


def test_build_client_uses_configured_seed() -> None:
    from src.common.config import TrainingConfig
    from src.fl.orchestrator import FlowerSmokeTestModel

    orchestrator = FedMedOrchestrator()

    orchestrator._training_config = TrainingConfig(
        local_epochs=2,
        batch_size=4,
        learning_rate=0.01,
        optimizer="sgd",
        seed=123,
    )

    client = orchestrator.build_client(
        "client_0",
        partition_index=0,
    )

    torch.manual_seed(123)
    expected_model = FlowerSmokeTestModel(
        name="configured_model_client_0",
        device="cpu",
    )

    for actual, expected in zip(
        client.get_parameters(),
        expected_model.get_parameters(),
    ):
        assert torch.equal(
            torch.from_numpy(actual),
            torch.from_numpy(expected),
        )


# ======================================================================
# Federated configuration wiring
# ======================================================================


def _capture_create_server_app_call(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """Replace create_server_app with a spy that records its policy args."""

    captured: dict[str, object] = {}

    def fake_create_server_app(
        initial_parameters_factory,
        strategy_factory=None,
        *,
        num_rounds=1,
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_available_nodes=1,
    ):
        captured["num_rounds"] = num_rounds
        captured["fraction_train"] = fraction_train
        captured["fraction_evaluate"] = fraction_evaluate
        captured["min_available_nodes"] = min_available_nodes
        return object()

    monkeypatch.setattr(
        "src.fl.orchestrator.create_server_app",
        fake_create_server_app,
    )

    return captured


def test_orchestrator_reads_federated_config() -> None:
    """The orchestrator must own the centralized federated config."""

    from src.common.config import FederatedConfig

    orchestrator = FedMedOrchestrator()

    assert isinstance(
        orchestrator._federated_config,
        FederatedConfig,
    )


def test_build_server_app_passes_federated_config_to_server_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator must forward the federated config into the runtime."""

    captured = _capture_create_server_app_call(monkeypatch)

    orchestrator = FedMedOrchestrator()

    result = orchestrator.build_server_app()

    assert result is not None

    config = orchestrator._federated_config

    assert captured == {
        "num_rounds": config.num_rounds,
        "fraction_train": config.fraction_fit,
        "fraction_evaluate": config.fraction_evaluate,
        "min_available_nodes": config.min_available_clients,
    }


def test_changing_federated_config_changes_runtime_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Changing the centralized federated configuration must change the
    runtime policy handed to the Flower server adapter.
    """

    from src.common.config import FederatedConfig

    captured = _capture_create_server_app_call(monkeypatch)

    orchestrator = FedMedOrchestrator()

    orchestrator._federated_config = FederatedConfig(
        strategy="fedavg",
        num_rounds=7,
        min_clients=2,
        min_available_clients=3,
        fraction_fit=0.5,
        fraction_evaluate=0.25,
    )

    orchestrator.build_server_app()

    assert captured == {
        "num_rounds": 7,
        "fraction_train": 0.5,
        "fraction_evaluate": 0.25,
        "min_available_nodes": 3,
    }


@pytest.mark.parametrize(
    ("federated_config", "error_message"),
    [
        (
            {
                "strategy": "fedavg",
                "num_rounds": 0,
                "min_clients": 1,
                "min_available_clients": 1,
                "fraction_fit": 1.0,
                "fraction_evaluate": 1.0,
            },
            "num_rounds must be a positive integer",
        ),
        (
            {
                "strategy": "fedavg",
                "num_rounds": 1,
                "min_clients": 1,
                "min_available_clients": 0,
                "fraction_fit": 1.0,
                "fraction_evaluate": 1.0,
            },
            "min_available_nodes must be a positive integer",
        ),
        (
            {
                "strategy": "fedavg",
                "num_rounds": 1,
                "min_clients": 1,
                "min_available_clients": 1,
                "fraction_fit": 0.0,
                "fraction_evaluate": 1.0,
            },
            "fraction_train must be a number in the range",
        ),
        (
            {
                "strategy": "fedavg",
                "num_rounds": 1,
                "min_clients": 1,
                "min_available_clients": 1,
                "fraction_fit": 1.0,
                "fraction_evaluate": 0.0,
            },
            "fraction_evaluate must be a number in the range",
        ),
    ],
)
def test_build_server_app_rejects_invalid_federated_config(
    federated_config: dict[str, object],
    error_message: str,
) -> None:
    """Invalid federated configuration must fail at the runtime boundary."""

    from src.common.config import FederatedConfig

    orchestrator = FedMedOrchestrator()
    orchestrator._federated_config = FederatedConfig(
        **federated_config,
    )

    with pytest.raises(
        FederatedLearningError,
        match=error_message,
    ):
        orchestrator.build_server_app()
