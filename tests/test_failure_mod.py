"""Tests for controlled Flower client failure injection."""

from flwr.app import (
    ConfigRecord,
    Context,
    Message,
    MessageType,
    RecordDict,
)

from app.failure_mod import e6_dropout_mod


def _call_next(message, context):
    return "SUCCESS"


def _context(
    partition_id: int,
    *,
    dropout_partition: int = 0,
    dropout_round: int = 2,
) -> Context:
    return Context(
        run_id=1,
        node_id=999999,
        node_config={
            "partition-id": partition_id,
            "num-partitions": 4,
        },
        state=RecordDict(),
        run_config={
            "e6-dropout-partition": dropout_partition,
            "e6-dropout-round": dropout_round,
        },
    )


def _message(round_number: int) -> Message:
    return Message(
        content=RecordDict({
            "fitins.config": ConfigRecord({
                "server-round": round_number,
            }),
        }),
        message_type=MessageType.TRAIN,
        dst_node_id=999999,
    )


def test_dropout_is_not_triggered_for_other_partition() -> None:
    result = e6_dropout_mod(
        _message(2),
        _context(1),
        _call_next,
    )

    assert result == "SUCCESS"


def test_dropout_is_not_triggered_for_other_round() -> None:
    result = e6_dropout_mod(
        _message(1),
        _context(0),
        _call_next,
    )

    assert result == "SUCCESS"


def test_dropout_is_triggered_for_matching_partition_and_round() -> None:
    try:
        e6_dropout_mod(
            _message(2),
            _context(0),
            _call_next,
        )
    except RuntimeError as exc:
        assert "E6 controlled client dropout" in str(exc)
    else:
        raise AssertionError(
            "Expected E6 dropout was not triggered"
        )
