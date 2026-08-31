"""
Tests for FedMed aggregation abstractions and FedAvg.

Phase 3.3-B coverage:

- Aggregator abstraction
- FedAvg sample-count weighting
- exact weighted-average mathematics
- single-client aggregation
- multi-client aggregation
- parameter structure validation
- parameter shape validation
- parameter dtype validation
- sample-count validation
- non-finite parameter rejection
- non-floating model-state handling
- input immutability
- output independence
- numerical stability
- invalid aggregation inputs

These tests intentionally use the existing Phase 3.2
FederatedFitResult and Phase 3.1 ParameterPayload contracts.

The tests do not depend on PyTorch models because mathematical
aggregation should be testable independently of model execution.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from src.common.exceptions import FederatedLearningError
from src.fl.aggregation import Aggregator
from src.fl.client import FederatedFitResult
from src.aggregation.fedavg import FedAvgAggregator


# ============================================================
# Helpers
# ============================================================


def make_fit_result(
    parameters: list[np.ndarray],
    *,
    num_examples: int = 1,
) -> FederatedFitResult:
    """
    Construct a minimal valid FederatedFitResult.

    Only parameters and num_examples affect FedAvg mathematics,
    but the complete Phase 3.2 result contract is populated.
    """

    return FederatedFitResult(
        parameters=parameters,
        num_examples=num_examples,
        metrics={
            "train_loss": 0.25,
        },
        epochs_completed=1,
        batches_processed=1,
        final_loss=0.25,
    )


def make_results(
    parameter_values: list[float],
    sample_counts: list[int],
) -> dict[str, FederatedFitResult]:
    """
    Create one scalar floating-point parameter per client.

    Example:

        parameter_values=[2.0, 6.0]
        sample_counts=[100, 300]

    represents:

        client_1 -> [2.0], n=100
        client_2 -> [6.0], n=300
    """

    assert len(parameter_values) == len(sample_counts)

    return {
        f"client_{index + 1}": make_fit_result(
            [
                np.array(
                    [value],
                    dtype=np.float32,
                )
            ],
            num_examples=sample_count,
        )
        for index, (
            value,
            sample_count,
        ) in enumerate(
            zip(
                parameter_values,
                sample_counts,
            )
        )
    }


# ============================================================
# Aggregator abstraction
# ============================================================


def test_aggregator_is_abstract() -> None:
    """
    Aggregator must remain an abstraction and cannot be
    instantiated directly.
    """

    with pytest.raises(TypeError):
        Aggregator()


def test_fedavg_is_an_aggregator() -> None:
    """
    FedAvgAggregator must implement the generic Aggregator
    contract.
    """

    aggregator = FedAvgAggregator()

    assert isinstance(
        aggregator,
        Aggregator,
    )


def test_fedavg_repr_is_informative() -> None:
    """Aggregator representation should identify its concrete type."""

    aggregator = FedAvgAggregator()

    assert repr(aggregator) == "FedAvgAggregator()"


# ============================================================
# Basic FedAvg mathematics
# ============================================================


def test_single_client_returns_equivalent_parameters() -> None:
    """
    With one successful client, FedAvg must return that client's
    parameters unchanged in value.
    """

    parameters = [
        np.array(
            [1.5, -2.0, 3.25],
            dtype=np.float32,
        ),
        np.array(
            [[4.0, 5.0]],
            dtype=np.float32,
        ),
    ]

    results = {
        "client_a": make_fit_result(
            parameters,
            num_examples=100,
        )
    }

    aggregated = FedAvgAggregator().aggregate(
        results,
    )

    assert len(aggregated) == len(parameters)

    for actual, expected in zip(
        aggregated,
        parameters,
    ):
        np.testing.assert_array_equal(
            actual,
            expected,
        )


def test_equal_sample_counts_produce_simple_average() -> None:
    """
    Equal sample counts must reduce FedAvg to the arithmetic mean.
    """

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [2.0, 4.0],
                    dtype=np.float32,
                )
            ],
            num_examples=100,
        ),
        "client_b": make_fit_result(
            [
                np.array(
                    [4.0, 8.0],
                    dtype=np.float32,
                )
            ],
            num_examples=100,
        ),
    }

    aggregated = FedAvgAggregator().aggregate(
        results,
    )

    np.testing.assert_allclose(
        aggregated[0],
        np.array(
            [3.0, 6.0],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_sample_count_weighted_fedavg() -> None:
    """
    Verify the canonical FedAvg equation:

        W = sum(n_k / N * W_k)
    """

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [2.0, 4.0],
                    dtype=np.float32,
                )
            ],
            num_examples=100,
        ),
        "client_b": make_fit_result(
            [
                np.array(
                    [6.0, 8.0],
                    dtype=np.float32,
                )
            ],
            num_examples=300,
        ),
    }

    aggregated = FedAvgAggregator().aggregate(
        results,
    )

    # 100 / 400 = 0.25
    # 300 / 400 = 0.75
    #
    # [2,4] * .25 + [6,8] * .75
    # = [5,7]
    expected = np.array(
        [5.0, 7.0],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        aggregated[0],
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_three_client_weighted_fedavg() -> None:
    """Verify weighted aggregation across three clients."""

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [1.0, 10.0],
                    dtype=np.float32,
                )
            ],
            num_examples=100,
        ),
        "client_b": make_fit_result(
            [
                np.array(
                    [3.0, 20.0],
                    dtype=np.float32,
                )
            ],
            num_examples=200,
        ),
        "client_c": make_fit_result(
            [
                np.array(
                    [5.0, 30.0],
                    dtype=np.float32,
                )
            ],
            num_examples=700,
        ),
    }

    aggregated = FedAvgAggregator().aggregate(
        results,
    )

    expected = (
        0.10 * np.array(
            [1.0, 10.0],
            dtype=np.float32,
        )
        + 0.20 * np.array(
            [3.0, 20.0],
            dtype=np.float32,
        )
        + 0.70 * np.array(
            [5.0, 30.0],
            dtype=np.float32,
        )
    )

    np.testing.assert_allclose(
        aggregated[0],
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_parameter_arrays_are_aggregated_independently() -> None:
    """Every model-state entry must be aggregated independently."""

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [1.0, 2.0],
                    dtype=np.float32,
                ),
                np.array(
                    [[10.0, 20.0]],
                    dtype=np.float32,
                ),
            ],
            num_examples=1,
        ),
        "client_b": make_fit_result(
            [
                np.array(
                    [3.0, 6.0],
                    dtype=np.float32,
                ),
                np.array(
                    [[30.0, 40.0]],
                    dtype=np.float32,
                ),
            ],
            num_examples=3,
        ),
    }

    aggregated = FedAvgAggregator().aggregate(
        results,
    )

    np.testing.assert_allclose(
        aggregated[0],
        np.array(
            [2.5, 5.0],
            dtype=np.float32,
        ),
    )

    np.testing.assert_allclose(
        aggregated[1],
        np.array(
            [[25.0, 35.0]],
            dtype=np.float32,
        ),
    )


# ============================================================
# Sample-count validation
# ============================================================


@pytest.mark.parametrize(
    "num_examples",
    [
        0,
        -1,
        -100,
    ],
)
def test_non_positive_sample_count_is_rejected(
    num_examples: int,
) -> None:
    """FedAvg requires every contributing client to have samples."""

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [1.0],
                    dtype=np.float32,
                )
            ],
            num_examples=num_examples,
        )
    }

    with pytest.raises(
        FederatedLearningError,
        match="num_examples",
    ):
        FedAvgAggregator().aggregate(
            results,
        )


@pytest.mark.parametrize(
    "num_examples",
    [
        1.5,
        "100",
        None,
        True,
        False,
    ],
)
def test_invalid_sample_count_type_is_rejected(
    num_examples,
) -> None:
    """FedAvg requires an integer sample count."""

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [1.0],
                    dtype=np.float32,
                )
            ],
            num_examples=num_examples,
        )
    }

    with pytest.raises(
        FederatedLearningError,
        match="num_examples",
    ):
        FedAvgAggregator().aggregate(
            results,
        )


def test_sample_weight_does_not_depend_on_metrics() -> None:
    """
    Parameter aggregation must use num_examples, not local
    accuracy/loss values.
    """

    result_a = FederatedFitResult(
        parameters=[
            np.array(
                [0.0],
                dtype=np.float32,
            )
        ],
        num_examples=100,
        metrics={
            "accuracy": 1.0,
        },
        epochs_completed=1,
        batches_processed=1,
        final_loss=0.01,
    )

    result_b = FederatedFitResult(
        parameters=[
            np.array(
                [10.0],
                dtype=np.float32,
            )
        ],
        num_examples=100,
        metrics={
            "accuracy": 0.0,
        },
        epochs_completed=1,
        batches_processed=1,
        final_loss=10.0,
    )

    aggregated = FedAvgAggregator().aggregate(
        {
            "a": result_a,
            "b": result_b,
        }
    )

    np.testing.assert_allclose(
        aggregated[0],
        np.array(
            [5.0],
            dtype=np.float32,
        ),
    )


# ============================================================
# Empty / malformed inputs
# ============================================================


def test_empty_results_are_rejected() -> None:
    """Aggregation without successful clients must fail."""

    with pytest.raises(
        FederatedLearningError,
        match="empty",
    ):
        FedAvgAggregator().aggregate(
            {},
        )


@pytest.mark.parametrize(
    "results",
    [
        None,
        [],
        (),
        "invalid",
        123,
    ],
)
def test_non_mapping_results_are_rejected(
    results,
) -> None:
    """Aggregator requires the canonical Mapping input."""

    with pytest.raises(
        FederatedLearningError,
        match="Mapping",
    ):
        FedAvgAggregator().aggregate(
            results,
        )


def test_non_string_client_id_is_rejected() -> None:
    """Client IDs must be strings."""

    results = {
        123: make_fit_result(
            [
                np.array(
                    [1.0],
                    dtype=np.float32,
                )
            ],
            num_examples=1,
        )
    }

    with pytest.raises(
        FederatedLearningError,
        match="keys must be strings",
    ):
        FedAvgAggregator().aggregate(
            results,
        )


def test_empty_client_id_is_rejected() -> None:
    """Empty client IDs are invalid."""

    results = {
        "   ": make_fit_result(
            [
                np.array(
                    [1.0],
                    dtype=np.float32,
                )
            ],
            num_examples=1,
        )
    }

    with pytest.raises(
        FederatedLearningError,
        match="empty.*client ID",
    ):
        FedAvgAggregator().aggregate(
            results,
        )


def test_invalid_result_object_is_rejected() -> None:
    """Values must be FederatedFitResult instances."""

    results = {
        "client_a": object(),
    }

    with pytest.raises(
        FederatedLearningError,
        match="FederatedFitResult",
    ):
        FedAvgAggregator().aggregate(
            results,
        )


# ============================================================
# Parameter validation
# ============================================================


def test_parameter_count_mismatch_is_rejected() -> None:
    """Clients must provide identical parameter counts."""

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [1.0],
                    dtype=np.float32,
                )
            ],
            num_examples=1,
        ),
        "client_b": make_fit_result(
            [
                np.array(
                    [2.0],
                    dtype=np.float32,
                ),
                np.array(
                    [3.0],
                    dtype=np.float32,
                ),
            ],
            num_examples=1,
        ),
    }

    with pytest.raises(
        FederatedLearningError,
        match="parameter count mismatch",
    ):
        FedAvgAggregator().aggregate(
            results,
        )


def test_parameter_shape_mismatch_is_rejected() -> None:
    """Clients must provide identical parameter shapes."""

    results = {
        "client_a": make_fit_result(
            [
                np.zeros(
                    (2, 3),
                    dtype=np.float32,
                )
            ],
            num_examples=1,
        ),
        "client_b": make_fit_result(
            [
                np.zeros(
                    (2, 4),
                    dtype=np.float32,
                )
            ],
            num_examples=1,
        ),
    }

    with pytest.raises(
        FederatedLearningError,
        match="shape mismatch",
    ):
        FedAvgAggregator().aggregate(
            results,
        )


def test_parameter_dtype_mismatch_is_rejected() -> None:
    """Clients must provide compatible parameter dtypes."""

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [1.0],
                    dtype=np.float32,
                )
            ],
            num_examples=1,
        ),
        "client_b": make_fit_result(
            [
                np.array(
                    [2.0],
                    dtype=np.float64,
                )
            ],
            num_examples=1,
        ),
    }

    with pytest.raises(
        FederatedLearningError,
        match="dtype mismatch",
    ):
        FedAvgAggregator().aggregate(
            results,
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_non_finite_parameter_is_rejected(
    bad_value: float,
) -> None:
    """NaN and infinite values must never enter global aggregation."""

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [bad_value],
                    dtype=np.float32,
                )
            ],
            num_examples=1,
        ),
        "client_b": make_fit_result(
            [
                np.array(
                    [1.0],
                    dtype=np.float32,
                )
            ],
            num_examples=1,
        ),
    }

    with pytest.raises(
        FederatedLearningError,
        match="non-finite",
    ):
        FedAvgAggregator().aggregate(
            results,
        )


def test_empty_parameter_payload_is_rejected() -> None:
    """A client cannot contribute an empty model state."""

    results = {
        "client_a": make_fit_result(
            [],
            num_examples=1,
        )
    }

    with pytest.raises(
        FederatedLearningError,
        match="empty parameter",
    ):
        FedAvgAggregator().aggregate(
            results,
        )


def test_non_numpy_parameter_is_rejected() -> None:
    """Every parameter entry must be a NumPy array."""

    results = {
        "client_a": make_fit_result(
            [
                [1.0, 2.0],
            ],
            num_examples=1,
        )
    }

    with pytest.raises(
        FederatedLearningError,
        match="NumPy array",
    ):
        FedAvgAggregator().aggregate(
            results,
        )


# ============================================================
# Non-floating state
# ============================================================


def test_identical_integer_state_is_preserved() -> None:
    """
    Identical non-floating model state is preserved rather than
    mathematically averaged.
    """

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [1.0, 2.0],
                    dtype=np.float32,
                ),
                np.array(
                    [10, 20],
                    dtype=np.int64,
                ),
            ],
            num_examples=100,
        ),
        "client_b": make_fit_result(
            [
                np.array(
                    [3.0, 4.0],
                    dtype=np.float32,
                ),
                np.array(
                    [10, 20],
                    dtype=np.int64,
                ),
            ],
            num_examples=300,
        ),
    }

    aggregated = FedAvgAggregator().aggregate(
        results,
    )

    np.testing.assert_allclose(
        aggregated[0],
        np.array(
            [2.5, 3.5],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        aggregated[1],
        np.array(
            [10, 20],
            dtype=np.int64,
        ),
    )

    assert aggregated[1].dtype == np.dtype(
        np.int64,
    )


def test_conflicting_integer_state_is_rejected() -> None:
    """
    Non-floating state must not be averaged when clients disagree.
    """

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [1.0],
                    dtype=np.float32,
                ),
                np.array(
                    [10],
                    dtype=np.int64,
                ),
            ],
            num_examples=1,
        ),
        "client_b": make_fit_result(
            [
                np.array(
                    [2.0],
                    dtype=np.float32,
                ),
                np.array(
                    [20],
                    dtype=np.int64,
                ),
            ],
            num_examples=1,
        ),
    }

    with pytest.raises(
        FederatedLearningError,
        match="non-floating",
    ):
        FedAvgAggregator().aggregate(
            results,
        )


def test_boolean_state_is_preserved_when_identical() -> None:
    """Identical boolean state is treated as discrete state."""

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [True, False],
                    dtype=np.bool_,
                )
            ],
            num_examples=10,
        ),
        "client_b": make_fit_result(
            [
                np.array(
                    [True, False],
                    dtype=np.bool_,
                )
            ],
            num_examples=20,
        ),
    }

    aggregated = FedAvgAggregator().aggregate(
        results,
    )

    np.testing.assert_array_equal(
        aggregated[0],
        np.array(
            [True, False],
            dtype=np.bool_,
        ),
    )


# ============================================================
# Immutability / defensive ownership
# ============================================================


def test_client_inputs_are_not_mutated() -> None:
    """Aggregation must never modify client-owned arrays."""

    client_a = [
        np.array(
            [1.0, 2.0],
            dtype=np.float32,
        )
    ]

    client_b = [
        np.array(
            [5.0, 6.0],
            dtype=np.float32,
        )
    ]

    original_a = client_a[0].copy()
    original_b = client_b[0].copy()

    results = {
        "client_a": make_fit_result(
            client_a,
            num_examples=1,
        ),
        "client_b": make_fit_result(
            client_b,
            num_examples=1,
        ),
    }

    FedAvgAggregator().aggregate(
        results,
    )

    np.testing.assert_array_equal(
        client_a[0],
        original_a,
    )

    np.testing.assert_array_equal(
        client_b[0],
        original_b,
    )


def test_output_is_independent_from_client_inputs() -> None:
    """
    Mutating the returned global parameter must not mutate the
    client parameters.
    """

    client_a = [
        np.array(
            [1.0],
            dtype=np.float32,
        )
    ]

    client_b = [
        np.array(
            [3.0],
            dtype=np.float32,
        )
    ]

    results = {
        "client_a": make_fit_result(
            client_a,
            num_examples=1,
        ),
        "client_b": make_fit_result(
            client_b,
            num_examples=1,
        ),
    }

    aggregated = FedAvgAggregator().aggregate(
        results,
    )

    aggregated[0][0] = 999.0

    np.testing.assert_array_equal(
        client_a[0],
        np.array(
            [1.0],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        client_b[0],
        np.array(
            [3.0],
            dtype=np.float32,
        ),
    )


def test_aggregation_is_stateless_between_calls() -> None:
    """One round must not influence another round."""

    aggregator = FedAvgAggregator()

    first = aggregator.aggregate(
        make_results(
            [0.0, 10.0],
            [1, 1],
        )
    )

    second = aggregator.aggregate(
        make_results(
            [100.0, 200.0],
            [1, 1],
        )
    )

    np.testing.assert_allclose(
        first[0],
        np.array(
            [5.0],
            dtype=np.float32,
        ),
    )

    np.testing.assert_allclose(
        second[0],
        np.array(
            [150.0],
            dtype=np.float32,
        ),
    )


# ============================================================
# Numerical / edge behavior
# ============================================================


def test_large_sample_imbalance_respects_weighting() -> None:
    """
    A client with overwhelmingly more samples should dominate
    the weighted result.
    """

    results = make_results(
        [0.0, 100.0],
        [1, 999],
    )

    aggregated = FedAvgAggregator().aggregate(
        results,
    )

    expected = (
        1.0 / 1000.0 * 0.0
        + 999.0 / 1000.0 * 100.0
    )

    np.testing.assert_allclose(
        aggregated[0],
        np.array(
            [expected],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_multidimensional_parameters_are_aggregated() -> None:
    """FedAvg must work for arbitrary compatible tensor shapes."""

    results = {
        "client_a": make_fit_result(
            [
                np.ones(
                    (2, 3, 4),
                    dtype=np.float32,
                )
            ],
            num_examples=1,
        ),
        "client_b": make_fit_result(
            [
                np.full(
                    (2, 3, 4),
                    3.0,
                    dtype=np.float32,
                )
            ],
            num_examples=3,
        ),
    }

    aggregated = FedAvgAggregator().aggregate(
        results,
    )

    assert aggregated[0].shape == (
        2,
        3,
        4,
    )

    np.testing.assert_allclose(
        aggregated[0],
        np.full(
            (2, 3, 4),
            2.5,
            dtype=np.float32,
        ),
    )


def test_aggregated_dtype_matches_reference_dtype() -> None:
    """
    The output floating dtype must remain compatible with the
    client model contract.
    """

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [1.0],
                    dtype=np.float32,
                )
            ],
            num_examples=1,
        ),
        "client_b": make_fit_result(
            [
                np.array(
                    [3.0],
                    dtype=np.float32,
                )
            ],
            num_examples=1,
        ),
    }

    aggregated = FedAvgAggregator().aggregate(
        results,
    )

    assert aggregated[0].dtype == np.dtype(
        np.float32,
    )


def test_result_is_a_new_parameter_payload() -> None:
    """Aggregation must return a new list, not the client's list."""

    parameters = [
        np.array(
            [1.0],
            dtype=np.float32,
        )
    ]

    results = {
        "client_a": make_fit_result(
            parameters,
            num_examples=1,
        )
    }

    aggregated = FedAvgAggregator().aggregate(
        results,
    )

    assert aggregated is not parameters
    assert aggregated[0] is not parameters[0]


# ============================================================
# Mapping compatibility
# ============================================================


def test_mapping_proxy_is_supported() -> None:
    """
    Aggregator accepts Mapping rather than requiring a concrete
    dict, matching the frozen API.
    """

    from types import MappingProxyType

    results = MappingProxyType(
        make_results(
            [2.0, 6.0],
            [1, 3],
        )
    )

    aggregated = FedAvgAggregator().aggregate(
        results,
    )

    np.testing.assert_allclose(
        aggregated[0],
        np.array(
            [5.0],
            dtype=np.float32,
        ),
    )