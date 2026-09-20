import torch

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
