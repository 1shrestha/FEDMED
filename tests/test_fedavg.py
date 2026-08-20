"""
Tests for FedAvgAggregator.

Scope
-----
This file tests only the concrete FedAvg implementation:

    src/aggregation/fedavg.py

The generic Aggregator abstraction is tested separately in:

    tests/test_aggregation.py

The Strategy layer is tested separately in:

    tests/test_strategy.py


FedAvg contract under test
--------------------------

For clients k:

                    n_k
    w_k = -------------------------
                 sum(n_j)

and:

                     K
    W = sum (w_k * W_k)
                    k=1


The tests verify:

- sample-count-weighted averaging
- single-client behavior
- multiple-client behavior
- multiple parameter tensors
- arbitrary tensor shapes
- parameter ordering
- parameter count compatibility
- parameter shape compatibility
- parameter dtype compatibility
- positive sample-count requirements
- non-finite value rejection
- non-floating state preservation
- conflicting discrete state rejection
- input immutability
- output independence
- dtype preservation
- numerical behavior
- mapping compatibility
- aggregator statelessness
- defensive error boundaries

The tests intentionally use the existing FedMed
FederatedFitResult contract rather than introducing a separate
test-only result representation.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from src.aggregation.fedavg import FedAvgAggregator
from src.common.exceptions import FederatedLearningError
from src.fl.client import FederatedFitResult


# ============================================================
# Test helpers
# ============================================================


def make_fit_result(
    parameters: list[np.ndarray],
    *,
    num_examples: int = 1,
    metrics: dict[str, float] | None = None,
) -> FederatedFitResult:
    """
    Build a valid FederatedFitResult for aggregation tests.

    FedAvg uses:

        parameters
        num_examples

    The remaining fields are populated according to the existing
    Phase 3.2 result contract.
    """

    return FederatedFitResult(
        parameters=parameters,
        num_examples=num_examples,
        metrics=(
            {"train_loss": 0.25}
            if metrics is None
            else metrics
        ),
        epochs_completed=1,
        batches_processed=1,
        final_loss=0.25,
    )


def scalar_parameters(
    value: float,
    *,
    dtype: np.dtype | type = np.float32,
) -> list[np.ndarray]:
    """Create one scalar-like parameter vector."""

    return [
        np.array(
            [value],
            dtype=dtype,
        )
    ]


def make_scalar_results(
    values: list[float],
    sample_counts: list[int],
) -> dict[str, FederatedFitResult]:
    """
    Create simple scalar client results.

    Example:

        values=[2, 6]
        sample_counts=[100, 300]

    produces:

        client_1 -> parameter [2], n=100
        client_2 -> parameter [6], n=300
    """

    if len(values) != len(sample_counts):
        raise ValueError(
            "values and sample_counts must have "
            "the same length."
        )

    return {
        f"client_{index + 1}": make_fit_result(
            scalar_parameters(value),
            num_examples=num_examples,
        )
        for index, (
            value,
            num_examples,
        ) in enumerate(
            zip(
                values,
                sample_counts,
            )
        )
    }


# ============================================================
# Fixture
# ============================================================


@pytest.fixture
def aggregator() -> FedAvgAggregator:
    """Return a fresh FedAvgAggregator for each test."""

    return FedAvgAggregator()


# ============================================================
# Basic behavior
# ============================================================


def test_single_client_is_identity(
    aggregator: FedAvgAggregator,
) -> None:
    """
    A single client has weight 1.0.

    Therefore:

        global_model == client_model
    """

    parameters = [
        np.array(
            [1.0, 2.0, 3.0],
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

    aggregated = aggregator.aggregate(
        results,
    )

    assert len(aggregated) == 2

    for actual, expected in zip(
        aggregated,
        parameters,
    ):
        np.testing.assert_array_equal(
            actual,
            expected,
        )


def test_equal_sample_counts_produce_arithmetic_mean(
    aggregator: FedAvgAggregator,
) -> None:
    """
    When:

        n_1 = n_2

    FedAvg becomes:

        W = (W_1 + W_2) / 2
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
            num_examples=100,
        ),
    }

    aggregated = aggregator.aggregate(
        results,
    )

    np.testing.assert_allclose(
        aggregated[0],
        np.array(
            [4.0, 6.0],
            dtype=np.float32,
        ),
    )


def test_sample_count_weighting_is_correct(
    aggregator: FedAvgAggregator,
) -> None:
    """
    Canonical FedAvg example:

        client A:
            W_A = [2, 4]
            n_A = 100

        client B:
            W_B = [6, 8]
            n_B = 300

        W = 0.25 W_A + 0.75 W_B

          = [5, 7]
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

    aggregated = aggregator.aggregate(
        results,
    )

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


def test_three_client_weighted_average(
    aggregator: FedAvgAggregator,
) -> None:
    """Verify FedAvg with three clients and unequal datasets."""

    results = {
        "client_a": make_fit_result(
            scalar_parameters(1.0),
            num_examples=100,
        ),
        "client_b": make_fit_result(
            scalar_parameters(3.0),
            num_examples=200,
        ),
        "client_c": make_fit_result(
            scalar_parameters(5.0),
            num_examples=700,
        ),
    }

    aggregated = aggregator.aggregate(
        results,
    )

    # 0.10 * 1 + 0.20 * 3 + 0.70 * 5 = 4.2
    expected = np.array(
        [4.2],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        aggregated[0],
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_parameter_entries_are_aggregated_independently(
    aggregator: FedAvgAggregator,
) -> None:
    """
    Each parameter entry must be aggregated according to its
    corresponding position in the model parameter payload.
    """

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
                np.array(
                    [100.0],
                    dtype=np.float32,
                ),
            ],
            num_examples=1,
        ),
        "client_b": make_fit_result(
            [
                np.array(
                    [5.0, 6.0],
                    dtype=np.float32,
                ),
                np.array(
                    [[30.0, 40.0]],
                    dtype=np.float32,
                ),
                np.array(
                    [200.0],
                    dtype=np.float32,
                ),
            ],
            num_examples=3,
        ),
    }

    aggregated = aggregator.aggregate(
        results,
    )

    np.testing.assert_allclose(
        aggregated[0],
        np.array(
            [4.0, 5.0],
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

    np.testing.assert_allclose(
        aggregated[2],
        np.array(
            [175.0],
            dtype=np.float32,
        ),
    )


def test_parameter_order_is_preserved(
    aggregator: FedAvgAggregator,
) -> None:
    """
    FedAvg must never reorder model parameter entries.

    This is important because ParameterPayload relies on a
    deterministic model-state ordering.
    """

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [1.0],
                    dtype=np.float32,
                ),
                np.array(
                    [10.0],
                    dtype=np.float32,
                ),
                np.array(
                    [100.0],
                    dtype=np.float32,
                ),
            ],
            num_examples=1,
        ),
        "client_b": make_fit_result(
            [
                np.array(
                    [3.0],
                    dtype=np.float32,
                ),
                np.array(
                    [30.0],
                    dtype=np.float32,
                ),
                np.array(
                    [300.0],
                    dtype=np.float32,
                ),
            ],
            num_examples=1,
        ),
    }

    aggregated = aggregator.aggregate(
        results,
    )

    np.testing.assert_allclose(
        aggregated[0],
        [2.0],
    )

    np.testing.assert_allclose(
        aggregated[1],
        [20.0],
    )

    np.testing.assert_allclose(
        aggregated[2],
        [200.0],
    )


# ============================================================
# Sample-count validation
# ============================================================


@pytest.mark.parametrize(
    "num_examples",
    [
        0,
        -1,
        -10,
        -1000,
    ],
)
def test_non_positive_num_examples_are_rejected(
    aggregator: FedAvgAggregator,
    num_examples: int,
) -> None:
    """Every contributing client must have a positive sample count."""

    results = {
        "client_a": make_fit_result(
            scalar_parameters(1.0),
            num_examples=num_examples,
        )
    }

    with pytest.raises(
        FederatedLearningError,
        match="num_examples",
    ):
        aggregator.aggregate(
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
        [],
        {},
    ],
)
def test_invalid_num_examples_types_are_rejected(
    aggregator: FedAvgAggregator,
    num_examples,
) -> None:
    """
    Sample counts must be actual integers.

    bool is deliberately rejected even though bool subclasses int
    in Python.
    """

    results = {
        "client_a": make_fit_result(
            scalar_parameters(1.0),
            num_examples=num_examples,
        )
    }

    with pytest.raises(
        FederatedLearningError,
        match="num_examples",
    ):
        aggregator.aggregate(
            results,
        )


def test_large_sample_imbalance_is_respected(
    aggregator: FedAvgAggregator,
) -> None:
    """
    A client with 999 samples must dominate a client with only
    one sample.
    """

    results = make_scalar_results(
        values=[0.0, 100.0],
        sample_counts=[1, 999],
    )

    aggregated = aggregator.aggregate(
        results,
    )

    expected = np.array(
        [99.9],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        aggregated[0],
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


# ============================================================
# Input validation
# ============================================================


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
def test_non_mapping_input_is_rejected(
    aggregator: FedAvgAggregator,
    results,
) -> None:
    """FedAvg requires Mapping[str, FederatedFitResult]."""

    with pytest.raises(
        FederatedLearningError,
        match="Mapping",
    ):
        aggregator.aggregate(
            results,
        )


def test_empty_mapping_is_rejected(
    aggregator: FedAvgAggregator,
) -> None:
    """FedAvg cannot aggregate zero clients."""

    with pytest.raises(
        FederatedLearningError,
        match="empty",
    ):
        aggregator.aggregate(
            {},
        )


def test_non_string_client_id_is_rejected(
    aggregator: FedAvgAggregator,
) -> None:
    """Client identifiers must be strings."""

    results = {
        123: make_fit_result(
            scalar_parameters(1.0),
            num_examples=1,
        )
    }

    with pytest.raises(
        FederatedLearningError,
        match="keys must be strings",
    ):
        aggregator.aggregate(
            results,
        )


def test_empty_client_id_is_rejected(
    aggregator: FedAvgAggregator,
) -> None:
    """Whitespace-only client IDs are invalid."""

    results = {
        "   ": make_fit_result(
            scalar_parameters(1.0),
            num_examples=1,
        )
    }

    with pytest.raises(
        FederatedLearningError,
        match="empty.*client ID",
    ):
        aggregator.aggregate(
            results,
        )


def test_invalid_result_type_is_rejected(
    aggregator: FedAvgAggregator,
) -> None:
    """Aggregation values must be FederatedFitResult objects."""

    results = {
        "client_a": object(),
    }

    with pytest.raises(
        FederatedLearningError,
        match="FederatedFitResult",
    ):
        aggregator.aggregate(
            results,
        )


# ============================================================
# Parameter structure validation
# ============================================================


def test_empty_parameter_payload_is_rejected(
    aggregator: FedAvgAggregator,
) -> None:
    """A model cannot be aggregated when it has no parameter entries."""

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
        aggregator.aggregate(
            results,
        )


def test_parameter_count_mismatch_is_rejected(
    aggregator: FedAvgAggregator,
) -> None:
    """All clients must have the same number of parameter entries."""

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
        aggregator.aggregate(
            results,
        )


def test_parameter_shape_mismatch_is_rejected(
    aggregator: FedAvgAggregator,
) -> None:
    """Corresponding parameter entries must have identical shapes."""

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
        aggregator.aggregate(
            results,
        )


def test_parameter_dtype_mismatch_is_rejected(
    aggregator: FedAvgAggregator,
) -> None:
    """Corresponding parameters must use the same dtype."""

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
        aggregator.aggregate(
            results,
        )


def test_non_numpy_parameter_is_rejected(
    aggregator: FedAvgAggregator,
) -> None:
    """Each parameter entry must be a NumPy ndarray."""

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
        aggregator.aggregate(
            results,
        )


# ============================================================
# Non-finite values
# ============================================================


@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_non_finite_floating_values_are_rejected(
    aggregator: FedAvgAggregator,
    bad_value: float,
) -> None:
    """
    NaN and infinity must never be allowed into global
    parameter aggregation.
    """

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
            scalar_parameters(1.0),
            num_examples=1,
        ),
    }

    with pytest.raises(
        FederatedLearningError,
        match="non-finite",
    ):
        aggregator.aggregate(
            results,
        )


def test_non_finite_value_in_second_parameter_is_rejected(
    aggregator: FedAvgAggregator,
) -> None:
    """Validation must inspect every model-state entry."""

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [1.0],
                    dtype=np.float32,
                ),
                np.array(
                    [np.nan],
                    dtype=np.float32,
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
                    [3.0],
                    dtype=np.float32,
                ),
            ],
            num_examples=1,
        ),
    }

    with pytest.raises(
        FederatedLearningError,
        match="non-finite",
    ):
        aggregator.aggregate(
            results,
        )


# ============================================================
# Non-floating model state
# ============================================================


def test_identical_integer_state_is_preserved(
    aggregator: FedAvgAggregator,
) -> None:
    """
    Integer state must not be averaged.

    If all clients agree, the state is preserved exactly.
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
            num_examples=1,
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
            num_examples=3,
        ),
    }

    aggregated = aggregator.aggregate(
        results,
    )

    # Floating parameter:
    #
    # 0.25 * [1,2] + 0.75 * [3,4]
    # = [2.5,3.5]
    np.testing.assert_allclose(
        aggregated[0],
        np.array(
            [2.5, 3.5],
            dtype=np.float32,
        ),
    )

    # Integer state is preserved.
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


def test_conflicting_integer_state_is_rejected(
    aggregator: FedAvgAggregator,
) -> None:
    """
    Integer/discrete state cannot be meaningfully averaged.

    Therefore conflicting values must fail rather than silently
    producing an arbitrary integer.
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
                    [3.0],
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
        aggregator.aggregate(
            results,
        )


def test_identical_boolean_state_is_preserved(
    aggregator: FedAvgAggregator,
) -> None:
    """Boolean state is discrete and should be preserved exactly."""

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [True, False],
                    dtype=np.bool_,
                )
            ],
            num_examples=1,
        ),
        "client_b": make_fit_result(
            [
                np.array(
                    [True, False],
                    dtype=np.bool_,
                )
            ],
            num_examples=2,
        ),
    }

    aggregated = aggregator.aggregate(
        results,
    )

    np.testing.assert_array_equal(
        aggregated[0],
        np.array(
            [True, False],
            dtype=np.bool_,
        ),
    )


def test_conflicting_boolean_state_is_rejected(
    aggregator: FedAvgAggregator,
) -> None:
    """Conflicting boolean state must not be averaged."""

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [True, False],
                    dtype=np.bool_,
                )
            ],
            num_examples=1,
        ),
        "client_b": make_fit_result(
            [
                np.array(
                    [False, False],
                    dtype=np.bool_,
                )
            ],
            num_examples=1,
        ),
    }

    with pytest.raises(
        FederatedLearningError,
        match="non-floating",
    ):
        aggregator.aggregate(
            results,
        )


# ============================================================
# Input immutability
# ============================================================


def test_client_parameter_arrays_are_not_mutated(
    aggregator: FedAvgAggregator,
) -> None:
    """
    Aggregation must treat client parameters as read-only.
    """

    client_a_parameters = [
        np.array(
            [1.0, 2.0],
            dtype=np.float32,
        )
    ]

    client_b_parameters = [
        np.array(
            [5.0, 6.0],
            dtype=np.float32,
        )
    ]

    original_a = client_a_parameters[0].copy()
    original_b = client_b_parameters[0].copy()

    results = {
        "client_a": make_fit_result(
            client_a_parameters,
            num_examples=1,
        ),
        "client_b": make_fit_result(
            client_b_parameters,
            num_examples=1,
        ),
    }

    aggregator.aggregate(
        results,
    )

    np.testing.assert_array_equal(
        client_a_parameters[0],
        original_a,
    )

    np.testing.assert_array_equal(
        client_b_parameters[0],
        original_b,
    )


def test_output_arrays_are_independent_from_inputs(
    aggregator: FedAvgAggregator,
) -> None:
    """
    Mutating the returned global model must never modify a client's
    local model.
    """

    client_a_parameters = [
        np.array(
            [1.0],
            dtype=np.float32,
        )
    ]

    client_b_parameters = [
        np.array(
            [3.0],
            dtype=np.float32,
        )
    ]

    results = {
        "client_a": make_fit_result(
            client_a_parameters,
            num_examples=1,
        ),
        "client_b": make_fit_result(
            client_b_parameters,
            num_examples=1,
        ),
    }

    aggregated = aggregator.aggregate(
        results,
    )

    aggregated[0][0] = 999.0

    np.testing.assert_array_equal(
        client_a_parameters[0],
        np.array(
            [1.0],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        client_b_parameters[0],
        np.array(
            [3.0],
            dtype=np.float32,
        ),
    )


def test_output_payload_is_a_new_list(
    aggregator: FedAvgAggregator,
) -> None:
    """The returned ParameterPayload must not be the client's list."""

    parameters = scalar_parameters(10.0)

    results = {
        "client_a": make_fit_result(
            parameters,
            num_examples=1,
        )
    }

    aggregated = aggregator.aggregate(
        results,
    )

    assert aggregated is not parameters
    assert aggregated[0] is not parameters[0]


# ============================================================
# Shape / dimensionality coverage
# ============================================================


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (10,),
        (2, 3),
        (3, 4, 5),
        (2, 3, 4, 5),
    ],
)
def test_arbitrary_parameter_shapes_are_supported(
    aggregator: FedAvgAggregator,
    shape: tuple[int, ...],
) -> None:
    """Compatible NumPy tensor shapes must aggregate correctly."""

    client_a = np.ones(
        shape,
        dtype=np.float32,
    )

    client_b = np.full(
        shape,
        3.0,
        dtype=np.float32,
    )

    results = {
        "client_a": make_fit_result(
            [client_a],
            num_examples=1,
        ),
        "client_b": make_fit_result(
            [client_b],
            num_examples=1,
        ),
    }

    aggregated = aggregator.aggregate(
        results,
    )

    assert aggregated[0].shape == shape

    np.testing.assert_allclose(
        aggregated[0],
        np.full(
            shape,
            2.0,
            dtype=np.float32,
        ),
    )


# ============================================================
# Dtype behavior
# ============================================================


@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.float32,
        np.float64,
    ],
)
def test_reference_floating_dtype_is_preserved(
    aggregator: FedAvgAggregator,
    dtype,
) -> None:
    """
    Accumulation may use a wider precision internally, but the
    final result must return to the model's reference dtype.
    """

    results = {
        "client_a": make_fit_result(
            [
                np.array(
                    [1.0],
                    dtype=dtype,
                )
            ],
            num_examples=1,
        ),
        "client_b": make_fit_result(
            [
                np.array(
                    [3.0],
                    dtype=dtype,
                )
            ],
            num_examples=1,
        ),
    }

    aggregated = aggregator.aggregate(
        results,
    )

    assert aggregated[0].dtype == np.dtype(
        dtype,
    )

    np.testing.assert_allclose(
        aggregated[0],
        np.array(
            [2.0],
            dtype=dtype,
        ),
        rtol=1e-3 if dtype == np.float16 else 1e-6,
        atol=1e-3 if dtype == np.float16 else 1e-6,
    )


# ============================================================
# Mapping compatibility
# ============================================================


def test_mapping_interface_is_supported(
    aggregator: FedAvgAggregator,
) -> None:
    """
    The frozen contract uses Mapping rather than requiring dict.
    """

    from types import MappingProxyType

    source = make_scalar_results(
        values=[2.0, 6.0],
        sample_counts=[1, 3],
    )

    results: Mapping[str, FederatedFitResult] = (
        MappingProxyType(source)
    )

    aggregated = aggregator.aggregate(
        results,
    )

    np.testing.assert_allclose(
        aggregated[0],
        np.array(
            [5.0],
            dtype=np.float32,
        ),
    )


# ============================================================
# Metric independence
# ============================================================


def test_metrics_do_not_affect_parameter_aggregation(
    aggregator: FedAvgAggregator,
) -> None:
    """
    FedAvg parameter weighting is based on num_examples.

    Metrics are intentionally not part of parameter aggregation.
    """

    result_a = make_fit_result(
        scalar_parameters(0.0),
        num_examples=100,
        metrics={
            "accuracy": 1.0,
            "train_loss": 0.01,
        },
    )

    result_b = make_fit_result(
        scalar_parameters(10.0),
        num_examples=100,
        metrics={
            "accuracy": 0.0,
            "train_loss": 100.0,
        },
    )

    aggregated = aggregator.aggregate(
        {
            "client_a": result_a,
            "client_b": result_b,
        }
    )

    # Metrics do not influence the 50/50 parameter average.
    np.testing.assert_allclose(
        aggregated[0],
        np.array(
            [5.0],
            dtype=np.float32,
        ),
    )


# ============================================================
# Statelessness
# ============================================================


def test_aggregator_is_stateless_between_rounds(
    aggregator: FedAvgAggregator,
) -> None:
    """
    A reusable FedAvgAggregator must not retain model state,
    client state, weights, or previous round results.
    """

    first = aggregator.aggregate(
        make_scalar_results(
            values=[0.0, 10.0],
            sample_counts=[1, 1],
        )
    )

    second = aggregator.aggregate(
        make_scalar_results(
            values=[100.0, 200.0],
            sample_counts=[1, 1],
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
# Numerical behavior
# ============================================================


def test_weighted_result_is_within_client_parameter_range(
    aggregator: FedAvgAggregator,
) -> None:
    """
    For positive FedAvg weights summing to one, the scalar result
    must lie within the participating scalar parameter range.
    """

    results = make_scalar_results(
        values=[
            -100.0,
            20.0,
            500.0,
        ],
        sample_counts=[
            10,
            20,
            30,
        ],
    )

    aggregated = aggregator.aggregate(
        results,
    )

    value = float(
        aggregated[0][0]
    )

    assert -100.0 <= value <= 500.0


def test_all_clients_with_same_parameters_return_same_parameters(
    aggregator: FedAvgAggregator,
) -> None:
    """
    If every client has exactly the same model, the global model
    must remain exactly the same regardless of sample counts.
    """

    parameters_a = [
        np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float32,
        ),
        np.array(
            [[4.0, 5.0]],
            dtype=np.float32,
        ),
    ]

    parameters_b = [
        array.copy()
        for array in parameters_a
    ]

    parameters_c = [
        array.copy()
        for array in parameters_a
    ]

    results = {
        "client_a": make_fit_result(
            parameters_a,
            num_examples=1,
        ),
        "client_b": make_fit_result(
            parameters_b,
            num_examples=100,
        ),
        "client_c": make_fit_result(
            parameters_c,
            num_examples=10000,
        ),
    }

    aggregated = aggregator.aggregate(
        results,
    )

    for actual, expected in zip(
        aggregated,
        parameters_a,
    ):
        np.testing.assert_array_equal(
            actual,
            expected,
        )


# ============================================================
# Error boundary behavior
# ============================================================


def test_error_is_fedmed_domain_exception(
    aggregator: FedAvgAggregator,
) -> None:
    """
    Invalid aggregation inputs should use the existing FedMed
    domain exception rather than leaking arbitrary exceptions.
    """

    with pytest.raises(
        FederatedLearningError,
    ):
        aggregator.aggregate(
            {},
        )