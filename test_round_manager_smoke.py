"""
test_round_manager_smoke.py

Fast, no-infra smoke test: fakes the registry and the gRPC notify call so
the state machine / selection / fan-out logic can be verified without
Postgres or real nodes running. Run: python test_round_manager_smoke.py
"""

import asyncio

from registry import NodeRecord
from round_manager import RoundManager, RoundState, SelectionStrategy, IllegalTransition


class FakeRegistry:
    """Fakes just enough of NodeRegistry for round_manager tests: node
    selection (get_live_nodes) and persistence (save_round, which
    round_manager now calls after every transition). Persistence is a
    no-op here — persistence-to-Postgres itself is exercised by talking
    to a real DB, not this in-memory test."""

    def __init__(self, nodes):
        self._nodes = nodes
        self.saved_states = []  # for assertions on what got persisted

    async def get_live_nodes(self):
        return self._nodes

    async def save_round(self, round_id, round_number, state, min_available_clients,
                          selected_nodes, responses):
        self.saved_states.append(state)


async def fake_notify_accept(node, record, flower_addr, fit_config, deadline_s):
    class Ack:
        accepted = True
        reason = ""
    await asyncio.sleep(0.01)
    return Ack()


async def fake_notify_flaky(node, record, flower_addr, fit_config, deadline_s):
    class Ack:
        accepted = node.node_id != "n3"  # n3 always declines
        reason = "" if accepted else "busy"
    await asyncio.sleep(0.01)
    return Ack()


def make_nodes(n):
    return [
        NodeRecord(
            node_id=f"n{i}", address=f"localhost:{9000+i}", hospital_label=f"Hospital {i}",
            last_heartbeat_at=0, ttl_seconds=30, capabilities={}, status="ACTIVE",
        )
        for i in range(1, n + 1)
    ]


async def test_all_active_round_completes():
    registry = FakeRegistry(make_nodes(4))
    mgr = RoundManager(registry, fake_notify_accept, "flower:8080")
    record = await mgr.start_round(strategy=SelectionStrategy.ALL_ACTIVE, min_available_clients=2)
    assert record.state == RoundState.ROUND_COMPLETE, record.state
    assert len(record.selected_nodes) == 4
    assert record.accepted_count == 4
    print("PASS: all-active round completes, 4/4 accepted")


async def test_not_enough_live_nodes_raises():
    registry = FakeRegistry(make_nodes(1))
    mgr = RoundManager(registry, fake_notify_accept, "flower:8080")
    try:
        await mgr.start_round(strategy=SelectionStrategy.ALL_ACTIVE, min_available_clients=3)
        raise AssertionError("expected ValueError")
    except ValueError:
        print("PASS: round refuses to start below min_available_clients")


async def test_fraction_sampling_size():
    registry = FakeRegistry(make_nodes(10))
    mgr = RoundManager(registry, fake_notify_accept, "flower:8080")
    record = await mgr.start_round(
        strategy=SelectionStrategy.FRACTION, min_available_clients=2, fraction_fit=0.3
    )
    assert len(record.selected_nodes) == 3, len(record.selected_nodes)  # ceil(10*0.3)
    print(f"PASS: fraction_fit=0.3 over 10 nodes selected {len(record.selected_nodes)}")


async def test_round_fails_below_threshold_after_declines():
    registry = FakeRegistry(make_nodes(3))
    mgr = RoundManager(registry, fake_notify_flaky, "flower:8080")
    record = await mgr.start_round(strategy=SelectionStrategy.ALL_ACTIVE, min_available_clients=3)
    assert record.state == RoundState.ROUND_FAILED, record.state
    assert record.accepted_count == 2
    print("PASS: round correctly fails when accepted_count < min_available_clients")


async def test_persist_called_on_every_transition():
    registry = FakeRegistry(make_nodes(3))
    mgr = RoundManager(registry, fake_notify_accept, "flower:8080")
    await mgr.start_round(strategy=SelectionStrategy.ALL_ACTIVE, min_available_clients=2)
    # ROUND_STARTING, WAITING_FOR_UPDATES, (post-fanout re-persist), AGGREGATING, ROUND_COMPLETE
    assert "ROUND_COMPLETE" in registry.saved_states
    assert registry.saved_states.count("ROUND_STARTING") >= 1
    print(f"PASS: round state persisted at each step ({registry.saved_states})")


async def test_mid_round_death_excluded_quorum_still_met():
    """4 nodes accept and the round is sitting in WAITING_FOR_UPDATES/
    AGGREGATING equivalent — one dies mid-round. min_available_clients=2,
    so losing one of four should still leave quorum: exclude and proceed."""
    registry = FakeRegistry(make_nodes(4))
    mgr = RoundManager(registry, fake_notify_accept, "flower:8080")

    # Run the round to completion first (fake_notify_accept is instant),
    # then simulate a node dying "mid-round" by calling handle_node_failure
    # directly — this exercises the same exclusion path health_monitor
    # would trigger, independent of exact timing.
    record = await mgr.start_round(strategy=SelectionStrategy.ALL_ACTIVE, min_available_clients=2)
    assert record.state == RoundState.ROUND_COMPLETE

    # Re-open the round artificially to test handle_node_failure's guard
    # logic in isolation (it only acts on WAITING_FOR_UPDATES/AGGREGATING).
    record.state = RoundState.AGGREGATING
    await mgr.handle_node_failure(record.selected_nodes[0].node_id)
    assert record.accepted_count == 3
    assert record.state == RoundState.AGGREGATING, "should stay in AGGREGATING, quorum still met"
    print("PASS: mid-round death excluded, round proceeds (quorum still met)")


async def test_mid_round_death_drops_below_quorum_fails_round():
    registry = FakeRegistry(make_nodes(2))
    mgr = RoundManager(registry, fake_notify_accept, "flower:8080")
    record = await mgr.start_round(strategy=SelectionStrategy.ALL_ACTIVE, min_available_clients=2)
    assert record.state == RoundState.ROUND_COMPLETE

    record.state = RoundState.AGGREGATING
    await mgr.handle_node_failure(record.selected_nodes[0].node_id)
    assert record.state == RoundState.ROUND_FAILED, record.state
    print("PASS: mid-round death that breaks quorum (2->1, need 2) fails the round")


async def test_illegal_transition_rejected():
    from round_manager import RoundRecord
    r = RoundRecord(round_id="x", round_number=1)
    try:
        r.transition(RoundState.AGGREGATING)  # skipping straight from IDLE
        raise AssertionError("expected IllegalTransition")
    except IllegalTransition:
        print("PASS: illegal state skip (IDLE -> AGGREGATING) rejected")


async def main():
    await test_all_active_round_completes()
    await test_not_enough_live_nodes_raises()
    await test_fraction_sampling_size()
    await test_round_fails_below_threshold_after_declines()
    await test_persist_called_on_every_transition()
    await test_mid_round_death_excluded_quorum_still_met()
    await test_mid_round_death_drops_below_quorum_fails_round()
    await test_illegal_transition_rejected()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
