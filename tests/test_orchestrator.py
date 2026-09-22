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
