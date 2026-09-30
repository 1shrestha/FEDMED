"""
test_retry_smoke.py

No-infra test for retry.py: verifies retryable gRPC errors get retried
up to max_attempts, non-retryable errors propagate immediately (no
wasted retries), and success on a later attempt returns cleanly.
Run: python test_retry_smoke.py
"""

import asyncio

import grpc

from retry import RetryExhausted, grpc_retry


class FakeRpcError(grpc.aio.AioRpcError):
    """retry.py's _is_retryable() does isinstance(exc, AioRpcError), so
    the fake has to actually subclass it — but the real __init__/__repr__
    reach into gRPC-internal metadata objects we don't want to construct.
    Skip super().__init__ and override __repr__/__str__ so logging the
    error doesn't blow up on missing internals; .code() is all retry.py
    actually needs to make its retry/no-retry decision."""

    def __init__(self, code: grpc.StatusCode):
        self._fake_code = code  # avoid clashing with the real class's slots

    def code(self):
        return self._fake_code

    def __repr__(self):
        return f"FakeRpcError({self._fake_code})"

    __str__ = __repr__


def _rpc_error(code: grpc.StatusCode) -> FakeRpcError:
    return FakeRpcError(code)


async def test_retries_transient_then_succeeds():
    attempts = {"n": 0}

    @grpc_retry(max_attempts=4, base_delay_s=0.01, max_delay_s=0.02)
    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _rpc_error(grpc.StatusCode.UNAVAILABLE)
        return "ok"

    result = await flaky()
    assert result == "ok"
    assert attempts["n"] == 3
    print(f"PASS: retried transient UNAVAILABLE and succeeded on attempt {attempts['n']}")


async def test_gives_up_after_max_attempts():
    attempts = {"n": 0}

    @grpc_retry(max_attempts=3, base_delay_s=0.01, max_delay_s=0.02)
    async def always_fails():
        attempts["n"] += 1
        raise _rpc_error(grpc.StatusCode.DEADLINE_EXCEEDED)

    try:
        await always_fails()
        raise AssertionError("expected RetryExhausted")
    except RetryExhausted as e:
        assert attempts["n"] == 3
        assert e.attempts == 3
        print("PASS: gives up after max_attempts, raises RetryExhausted")


async def test_non_retryable_error_propagates_immediately():
    attempts = {"n": 0}

    @grpc_retry(max_attempts=5, base_delay_s=0.01, max_delay_s=0.02)
    async def bad_request():
        attempts["n"] += 1
        raise _rpc_error(grpc.StatusCode.INVALID_ARGUMENT)

    try:
        await bad_request()
        raise AssertionError("expected RetryExhausted wrapping the non-retryable error")
    except RetryExhausted:
        assert attempts["n"] == 1, "should not retry a non-retryable error"
        print("PASS: non-retryable error (INVALID_ARGUMENT) fails fast, no wasted retries")


async def main():
    await test_retries_transient_then_succeeds()
    await test_gives_up_after_max_attempts()
    await test_non_retryable_error_propagates_immediately()
    print("\nAll retry smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
