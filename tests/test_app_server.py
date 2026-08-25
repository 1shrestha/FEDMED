"""
Tests for the Flower server runtime adapter.

These tests intentionally exercise the Flower boundary only.  The
framework-independent FedMed core remains covered by the existing core
test suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp import ServerApp

from app.server import (
    ARRAYS_KEY,
    CONFIG_KEY,
    METRICS_KEY,
    NUM_EXAMPLES_KEY,
    FedMedFlowerStrategy,
    create_server_app,
)
from src.aggregation.fedavg import FedAvgAggregator
from src.common.exceptions import FederatedLearningError
from src.fl.client import FederatedFitResult
from src.fl.parameters import ParameterPayload
from src.fl.strategy import FedAvgStrategy


def make_parameters() -> ParameterPayload:
    return [
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([3.0], dtype=np.float32),
    ]


def make_fedmed_strategy() -> FedAvgStrategy:
    return FedAvgStrategy(aggregator=FedAvgAggregator())


class FakeGrid:
    """Minimal grid contract required by the server adapter tests."""

    def __init__(self, node_ids: list[int]) -> None:
        self._node_ids = list(node_ids)

    def get_node_ids(self) -> list[int]:
        return list(self._node_ids)


def _reply_message(
    *,
    content: RecordDict,
    message_type: MessageType,
    node_id: int,
) -> Message:
    """Create a semantically valid Flower reply from a node."""
    request = Message(
        content=RecordDict(),
        dst_node_id=node_id,
        message_type=message_type,
    )
    return request.create_reply(content)


def make_train_reply(
    *,
    node_id: int,
    parameters: ParameterPayload,
    num_examples: int = 4,
    metrics: dict[str, float] | None = None,
) -> Message:
    """Create a successful Flower training reply."""

    metric_values: dict[str, Any] = {
        NUM_EXAMPLES_KEY: num_examples,
        "accuracy": 0.8 if metrics is None else metrics["accuracy"],
        "loss": 0.25 if metrics is None else metrics["loss"],
        "epochs_completed": 1,
        "batches_processed": 2,
        "final_loss": 0.25,
    }

    content = RecordDict(
        {
            ARRAYS_KEY: ArrayRecord.from_numpy_ndarrays(parameters),
            METRICS_KEY: MetricRecord(metric_values),
        }
    )

    return _reply_message(
        content=content,
        message_type=MessageType.TRAIN,
        node_id=node_id,
    )


def make_evaluate_reply(
    *,
    node_id: int,
    num_examples: int = 4,
    loss: float = 0.5,
    accuracy: float = 0.75,
) -> Message:
    """Create a successful Flower evaluation reply."""

    content = RecordDict(
        {
            METRICS_KEY: MetricRecord(
                {
                    NUM_EXAMPLES_KEY: num_examples,
                    "loss": loss,
                    "accuracy": accuracy,
                }
            )
        }
    )

    return _reply_message(
        content=content,
        message_type=MessageType.EVALUATE,
        node_id=node_id,
    )


def test_strategy_requires_fedmed_strategy() -> None:
    with pytest.raises(FederatedLearningError):
        FedMedFlowerStrategy(
            fedmed_strategy=object(),  # type: ignore[arg-type]
        )


def test_strategy_accepts_real_fedmed_strategy() -> None:
    strategy = make_fedmed_strategy()

    adapter = FedMedFlowerStrategy(
        fedmed_strategy=strategy,
    )

    assert adapter.fedmed_strategy is strategy


@pytest.mark.parametrize("value", [-0.1, 1.1, 2.0])
def test_invalid_selection_fraction_is_rejected(value: float) -> None:
    with pytest.raises(FederatedLearningError):
        FedMedFlowerStrategy(
            fedmed_strategy=make_fedmed_strategy(),
            fraction_train=value,
        )


def test_invalid_min_available_nodes_is_rejected() -> None:
    with pytest.raises(FederatedLearningError):
        FedMedFlowerStrategy(
            fedmed_strategy=make_fedmed_strategy(),
            min_available_nodes=0,
        )


def test_node_selection_is_deterministic() -> None:
    adapter = FedMedFlowerStrategy(
        fedmed_strategy=make_fedmed_strategy(),
        fraction_train=1.0,
    )
    grid = FakeGrid([4, 2, 3, 1])

    assert adapter._select_nodes(grid, 1.0) == [1, 2, 3, 4]
    assert adapter._select_nodes(grid, 1.0) == [1, 2, 3, 4]


def test_node_selection_respects_fraction() -> None:
    adapter = FedMedFlowerStrategy(
        fedmed_strategy=make_fedmed_strategy(),
        fraction_train=0.5,
        min_available_nodes=1,
    )

    assert adapter._select_nodes(FakeGrid([1, 2, 3, 4]), 0.5) == [1, 2]


def test_node_selection_respects_minimum_nodes() -> None:
    adapter = FedMedFlowerStrategy(
        fedmed_strategy=make_fedmed_strategy(),
        fraction_train=0.1,
        min_available_nodes=2,
    )

    assert adapter._select_nodes(FakeGrid([1, 2, 3, 4]), 0.1) == [1, 2]


def test_node_selection_rejects_insufficient_nodes() -> None:
    adapter = FedMedFlowerStrategy(
        fedmed_strategy=make_fedmed_strategy(),
        min_available_nodes=3,
    )

    with pytest.raises(FederatedLearningError, match="Insufficient Flower nodes"):
        adapter._select_nodes(FakeGrid([1, 2]), 1.0)


def test_configure_train_creates_messages_for_selected_nodes() -> None:
    adapter = FedMedFlowerStrategy(
        fedmed_strategy=make_fedmed_strategy(),
    )

    arrays = ArrayRecord.from_numpy_ndarrays(make_parameters())
    config = ConfigRecord({"local_epochs": 1})

    messages = list(
        adapter.configure_train(
            server_round=3,
            arrays=arrays,
            config=config,
            grid=FakeGrid([1, 2]),
        )
    )

    assert len(messages) == 2
    assert [message.metadata.dst_node_id for message in messages] == [1, 2]

    for message in messages:
        assert ARRAYS_KEY in message.content
        assert CONFIG_KEY in message.content
        assert message.content[CONFIG_KEY]["server-round"] == 3


def test_training_reply_converts_to_fedmed_fit_result() -> None:
    adapter = FedMedFlowerStrategy(
        fedmed_strategy=make_fedmed_strategy(),
    )

    parameters = make_parameters()
    message = make_train_reply(
        node_id=7,
        parameters=parameters,
        num_examples=12,
    )

    result = adapter._fit_result_from_message(message)

    assert isinstance(result, FederatedFitResult)
    assert result.num_examples == 12

    np.testing.assert_array_equal(result.parameters[0], parameters[0])
    np.testing.assert_array_equal(result.parameters[1], parameters[1])

    assert result.metrics["accuracy"] == pytest.approx(0.8)
    assert result.epochs_completed == 1
    assert result.batches_processed == 2
    assert result.final_loss == pytest.approx(0.25)


def test_training_reply_requires_arrays() -> None:
    adapter = FedMedFlowerStrategy(
        fedmed_strategy=make_fedmed_strategy(),
    )

    message = _reply_message(
        content=RecordDict(
            {
                METRICS_KEY: MetricRecord(
                    {NUM_EXAMPLES_KEY: 4}
                )
            }
        ),
        message_type=MessageType.TRAIN,
        node_id=1,
    )

    with pytest.raises(FederatedLearningError, match="missing 'arrays'"):
        adapter._fit_result_from_message(message)


def test_training_reply_requires_metrics() -> None:
    adapter = FedMedFlowerStrategy(
        fedmed_strategy=make_fedmed_strategy(),
    )

    message = _reply_message(
        content=RecordDict(
            {
                ARRAYS_KEY: ArrayRecord.from_numpy_ndarrays(
                    make_parameters()
                )
            }
        ),
        message_type=MessageType.TRAIN,
        node_id=1,
    )

    with pytest.raises(FederatedLearningError, match="missing 'metrics'"):
        adapter._fit_result_from_message(message)


def test_aggregate_train_delegates_to_fedmed_strategy() -> None:
    adapter = FedMedFlowerStrategy(
        fedmed_strategy=make_fedmed_strategy(),
    )

    replies = [
        make_train_reply(
            node_id=1,
            parameters=[np.array([[1.0]], dtype=np.float32)],
            num_examples=1,
        ),
        make_train_reply(
            node_id=2,
            parameters=[np.array([[3.0]], dtype=np.float32)],
            num_examples=3,
        ),
    ]

    arrays, metrics = adapter.aggregate_train(
        server_round=1,
        replies=replies,
    )

    assert arrays is not None
    assert metrics is not None

    aggregated = arrays.to_numpy_ndarrays()

    np.testing.assert_allclose(
        aggregated[0],
        np.array([[2.5]], dtype=np.float32),
    )
    assert metrics[NUM_EXAMPLES_KEY] == pytest.approx(4.0)


def test_aggregate_train_returns_none_without_successful_replies() -> None:
    adapter = FedMedFlowerStrategy(
        fedmed_strategy=make_fedmed_strategy(),
    )

    arrays, metrics = adapter.aggregate_train(
        server_round=1,
        replies=[],
    )

    assert arrays is None
    assert metrics is None


def test_aggregate_train_rejects_duplicate_client_replies() -> None:
    adapter = FedMedFlowerStrategy(
        fedmed_strategy=make_fedmed_strategy(),
    )

    message = make_train_reply(
        node_id=1,
        parameters=make_parameters(),
    )

    with pytest.raises(
        FederatedLearningError,
        match="Duplicate training reply",
    ):
        adapter.aggregate_train(
            server_round=1,
            replies=[message, message],
        )


def test_configure_evaluate_creates_messages() -> None:
    adapter = FedMedFlowerStrategy(
        fedmed_strategy=make_fedmed_strategy(),
    )

    arrays = ArrayRecord.from_numpy_ndarrays(make_parameters())
    config = ConfigRecord({"batch_size": 8})

    messages = list(
        adapter.configure_evaluate(
            server_round=2,
            arrays=arrays,
            config=config,
            grid=FakeGrid([10, 20]),
        )
    )

    assert len(messages) == 2
    assert [message.metadata.dst_node_id for message in messages] == [10, 20]

    for message in messages:
        assert ARRAYS_KEY in message.content
        assert CONFIG_KEY in message.content
        assert message.content[CONFIG_KEY]["server-round"] == 2


def test_aggregate_evaluate_uses_sample_weighting() -> None:
    adapter = FedMedFlowerStrategy(
        fedmed_strategy=make_fedmed_strategy(),
    )

    replies = [
        make_evaluate_reply(
            node_id=1,
            num_examples=2,
            loss=1.0,
            accuracy=0.5,
        ),
        make_evaluate_reply(
            node_id=2,
            num_examples=6,
            loss=0.0,
            accuracy=1.0,
        ),
    ]

    metrics = adapter.aggregate_evaluate(
        server_round=1,
        replies=replies,
    )

    assert metrics is not None
    assert metrics["accuracy"] == pytest.approx(0.875)
    assert metrics["loss"] == pytest.approx(0.25)
    assert metrics[NUM_EXAMPLES_KEY] == pytest.approx(8.0)


def test_aggregate_evaluate_returns_none_without_replies() -> None:
    adapter = FedMedFlowerStrategy(
        fedmed_strategy=make_fedmed_strategy(),
    )

    assert (
        adapter.aggregate_evaluate(
            server_round=1,
            replies=[],
        )
        is None
    )


def test_parameter_copy_is_defensive() -> None:
    parameters = make_parameters()
    copied = FedMedFlowerStrategy._copy_parameters(parameters)

    copied[0][0, 0] = 999.0

    assert parameters[0][0, 0] == 1.0


def test_create_server_app_requires_parameter_factory() -> None:
    with pytest.raises(FederatedLearningError):
        create_server_app(object())  # type: ignore[arg-type]


def test_create_server_app_requires_positive_rounds() -> None:
    with pytest.raises(FederatedLearningError):
        create_server_app(
            lambda context: make_parameters(),
            num_rounds=0,
        )


def test_create_server_app_returns_server_app() -> None:
    app = create_server_app(
        lambda context: make_parameters(),
        num_rounds=1,
    )

    assert isinstance(app, ServerApp)


def test_create_server_app_accepts_strategy_factory() -> None:
    app = create_server_app(
        lambda context: make_parameters(),
        strategy_factory=lambda context: make_fedmed_strategy(),
        num_rounds=2,
    )

    assert isinstance(app, ServerApp)


@pytest.mark.parametrize(
    "module_name",
    [
        "src.fl.server",
        "src.fl.rounds",
        "src.fl.strategy",
        "src.fl.aggregation",
        "src.aggregation.fedavg",
    ],
)
def test_fedmed_core_modules_do_not_import_flower(
    module_name: str,
) -> None:
    """Flower must remain outside the framework-independent FedMed core."""

    module_path = (
        Path(__file__).resolve().parents[1]
        / Path(module_name.replace(".", "/") + ".py")
    )

    source = module_path.read_text(encoding="utf-8")

    assert "import flwr" not in source
    assert "from flwr" not in source