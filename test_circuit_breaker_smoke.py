"""
test_circuit_breaker_smoke.py

No-infra test for circuit_breaker.py's CLOSED -> OPEN -> HALF_OPEN ->
CLOSED lifecycle. Run: python test_circuit_breaker_smoke.py
"""

import time

from circuit_breaker import CircuitBreakerConfig, CircuitOpenError, CircuitState, NodeCircuitBreakers


def test_opens_after_threshold_failures():
    cb = NodeCircuitBreakers(CircuitBreakerConfig(failure_threshold=3, reset_timeout_s=10))
    for _ in range(2):
        cb.before_call("nodeA")  # should not raise yet
        cb.record_failure("nodeA")
    assert cb.state_of("nodeA") == CircuitState.CLOSED, "should still be closed after 2/3 failures"

    cb.before_call("nodeA")
    cb.record_failure("nodeA")
    assert cb.state_of("nodeA") == CircuitState.OPEN, "should open on the 3rd consecutive failure"
    print("PASS: breaker opens after failure_threshold consecutive failures")


def test_fails_fast_while_open():
    cb = NodeCircuitBreakers(CircuitBreakerConfig(failure_threshold=1, reset_timeout_s=10))
    cb.before_call("nodeB")
    cb.record_failure("nodeB")
    assert cb.state_of("nodeB") == CircuitState.OPEN

    try:
        cb.before_call("nodeB")
        raise AssertionError("expected CircuitOpenError")
    except CircuitOpenError:
        print("PASS: before_call raises CircuitOpenError while OPEN (no RPC attempted)")


def test_half_open_probe_success_closes_circuit():
    cb = NodeCircuitBreakers(CircuitBreakerConfig(failure_threshold=1, reset_timeout_s=0.05))
    cb.before_call("nodeC")
    cb.record_failure("nodeC")
    assert cb.state_of("nodeC") == CircuitState.OPEN

    time.sleep(0.06)  # let reset_timeout_s elapse
    cb.before_call("nodeC")  # should transition to HALF_OPEN and allow the probe through
    assert cb.state_of("nodeC") == CircuitState.HALF_OPEN
    cb.record_success("nodeC")
    assert cb.state_of("nodeC") == CircuitState.CLOSED
    print("PASS: OPEN -> HALF_OPEN after reset_timeout_s -> CLOSED on successful probe")


def test_half_open_probe_failure_reopens_immediately():
    cb = NodeCircuitBreakers(CircuitBreakerConfig(failure_threshold=1, reset_timeout_s=0.05))
    cb.before_call("nodeD")
    cb.record_failure("nodeD")
    time.sleep(0.06)
    cb.before_call("nodeD")
    assert cb.state_of("nodeD") == CircuitState.HALF_OPEN
    cb.record_failure("nodeD")  # probe fails
    assert cb.state_of("nodeD") == CircuitState.OPEN
    print("PASS: failed HALF_OPEN probe goes straight back to OPEN")


def test_independent_per_node():
    cb = NodeCircuitBreakers(CircuitBreakerConfig(failure_threshold=1, reset_timeout_s=10))
    cb.before_call("nodeE")
    cb.record_failure("nodeE")
    assert cb.state_of("nodeE") == CircuitState.OPEN
    assert cb.state_of("nodeF") == CircuitState.CLOSED, "a flaky node must not affect other nodes"
    print("PASS: breakers are independent per node_id")


if __name__ == "__main__":
    test_opens_after_threshold_failures()
    test_fails_fast_while_open()
    test_half_open_probe_success_closes_circuit()
    test_half_open_probe_failure_reopens_immediately()
    test_independent_per_node()
    print("\nAll circuit breaker smoke tests passed.")
