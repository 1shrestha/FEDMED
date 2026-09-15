"""Controlled Flower client failure injection for E6 experiments."""

from __future__ import annotations

from collections.abc import Callable

from flwr.app import Context, Message, MessageType


def e6_dropout_mod(
    message: Message,
    context: Context,
    call_next: Callable[[Message, Context], Message],
) -> Message:
    """Inject a controlled training-client failure for E6.

    The failure occurs only when the configured partition and federated
    training round both match the incoming TRAIN message.

    Configuration:
        e6-dropout-partition:
            Stable Flower partition ID to fail.
        e6-dropout-round:
            Federated round in which the partition should fail.

    If either setting is absent, the modifier is a no-op.
    """

    if message.metadata.message_type != MessageType.TRAIN:
        return call_next(message, context)

    dropout_partition = context.run_config.get("e6-dropout-partition")
    dropout_round = context.run_config.get("e6-dropout-round")

    if dropout_partition is None or dropout_round is None:
        return call_next(message, context)

    try:
        target_partition = int(dropout_partition)
        target_round = int(dropout_round)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "E6 dropout configuration requires integer "
            "'e6-dropout-partition' and 'e6-dropout-round'."
        ) from exc

    partition_id = context.node_config.get("partition-id")

    try:
        partition_id = int(partition_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "E6 dropout requires an integer 'partition-id' "
            "in the Flower node configuration."
        ) from exc

    train_config = message.content.get("fitins.config", {})
    message_round = train_config.get("server-round")

    if (
        partition_id == target_partition
        and message_round == target_round
    ):
        print(
            "[FedMed E6] controlled client dropout: "
            f"partition={partition_id}, round={message_round}",
            flush=True,
        )
        raise RuntimeError(
            "E6 controlled client dropout: "
            f"partition={partition_id}, round={message_round}"
        )

    return call_next(message, context)


__all__ = ["e6_dropout_mod"]
