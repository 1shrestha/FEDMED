"""
security/tests/test_defenses.py
================================
Unit tests for security.defenses — Byzantine-robust aggregation.

Tests cover all four algorithms:
1. trimmed_mean
   - Basic correctness: result within honest range.
   - Poisoning scenario: outlier excluded from mean.
   - trim_ratio=0 degrades to plain mean.
   - Invalid trim_ratio raises ValueError.
   - Single update is returned as-is (trim=0).
   - Empty list raises ValueError.

2. coordinate_median
   - Basic correctness: median of symmetric updates = center.
   - Byzantine resilience: median is robust to large outlier.
   - Odd vs. even number of updates.
   - Empty list raises ValueError.

3. krum
   - Selects the honest update, not the Byzantine one.
   - Returns a 1-D array of the same shape as inputs.
   - Invalid f (2f+2 >= n) raises ValueError.
   - Empty list raises ValueError.

4. multi_krum
   - With m=1 behaves identically to krum.
   - With m=n-f returns a mean (but excludes f worst updates).
   - Poisoning: malicious update not included in selected set.
   - Default m = n - f.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from security.defenses import (
    coordinate_median,
    krum,
    multi_krum,
    trimmed_mean,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(seed=0)


def _honest(n: int, d: int, center: float = 0.0, noise: float = 0.01) -> list[np.ndarray]:
    """Generate n honest updates clustered around *center*."""
    return [
        np.full(d, center, dtype=np.float64) + RNG.standard_normal(d) * noise
        for _ in range(n)
    ]


def _with_byzantine(
    honest_updates: list[np.ndarray],
    magnitude: float = 1000.0,
) -> list[np.ndarray]:
    """Append one Byzantine update with a large outlier magnitude."""
    d = honest_updates[0].shape[0]
    byzantine = np.full(d, magnitude, dtype=np.float64)
    return honest_updates + [byzantine]


# ===========================================================================
# 1. Trimmed Mean
# ===========================================================================


class TestTrimmedMean(unittest.TestCase):
    def test_basic_correctness(self):
        """Without Byzantine clients, trimmed mean ≈ plain mean."""
        updates = [np.array([float(i)], dtype=np.float64) for i in range(10)]
        result = trimmed_mean(updates, trim_ratio=0.1)
        plain = np.mean([u[0] for u in updates])
        self.assertAlmostEqual(float(result[0]), plain, places=2)

    def test_trim_ratio_zero_equals_mean(self):
        """trim_ratio=0 must return the plain mean."""
        updates = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
        result = trimmed_mean(updates, trim_ratio=0.0)
        expected = np.mean(updates, axis=0)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_poisoning_excluded(self):
        """With trim_ratio=0.2, one Byzantine outlier of 10 clients is removed."""
        honest = _honest(9, d=5, center=0.0, noise=0.001)
        byzantine = [np.full(5, 1000.0)]
        updates = honest + byzantine  # 10 clients total

        result = trimmed_mean(updates, trim_ratio=0.1)  # removes 1 from each tail
        # Result should be very close to 0.0 (honest center), not near 1000
        self.assertLess(np.abs(result).max(), 1.0)

    def test_single_update_no_trim(self):
        """Single update with trim_ratio=0 returns that update."""
        u = np.array([3.14, 2.71])
        result = trimmed_mean([u], trim_ratio=0.0)
        np.testing.assert_allclose(result, u)

    def test_invalid_trim_ratio_high(self):
        with self.assertRaises(ValueError):
            trimmed_mean([np.ones(3)] * 5, trim_ratio=0.5)

    def test_invalid_trim_ratio_negative(self):
        with self.assertRaises(ValueError):
            trimmed_mean([np.ones(3)] * 5, trim_ratio=-0.1)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            trimmed_mean([], trim_ratio=0.1)

    def test_mismatched_shapes_raises(self):
        with self.assertRaises(ValueError):
            trimmed_mean([np.ones(3), np.ones(4)], trim_ratio=0.0)

    def test_all_same_updates(self):
        """Trimmed mean of identical updates must equal that update."""
        u = np.array([1.0, 2.0, 3.0])
        result = trimmed_mean([u.copy() for _ in range(6)], trim_ratio=0.1)
        np.testing.assert_allclose(result, u, rtol=1e-6)


# ===========================================================================
# 2. Coordinate-wise Median
# ===========================================================================


class TestCoordinateMedian(unittest.TestCase):
    def test_odd_number_of_updates(self):
        """Median of 5 updates must equal the middle update per coordinate."""
        updates = [np.array([float(i)]) for i in range(5)]
        result = coordinate_median(updates)
        self.assertAlmostEqual(float(result[0]), 2.0)

    def test_even_number_of_updates(self):
        """numpy.median for even n returns average of two middle values."""
        updates = [np.array([float(i)]) for i in range(4)]  # 0,1,2,3
        result = coordinate_median(updates)
        self.assertAlmostEqual(float(result[0]), 1.5)

    def test_byzantine_resilience(self):
        """Median is unaffected by a single massive outlier among 7 clients."""
        honest = _honest(6, d=10, center=1.0, noise=0.01)
        byzantine = [np.full(10, 1e6, dtype=np.float64)]
        result = coordinate_median(honest + byzantine)
        # Should still be very close to the honest centre of ~1.0
        np.testing.assert_allclose(result, np.ones(10), atol=0.1)

    def test_symmetry(self):
        """Median of symmetric updates centred at zero should be ≈ 0."""
        updates = [np.array([v]) for v in [-3.0, -1.0, 0.0, 1.0, 3.0]]
        result = coordinate_median(updates)
        self.assertAlmostEqual(float(result[0]), 0.0)

    def test_single_update_returned(self):
        u = np.array([5.0, 6.0])
        result = coordinate_median([u])
        np.testing.assert_array_equal(result, u)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            coordinate_median([])

    def test_mismatched_shapes_raises(self):
        with self.assertRaises(ValueError):
            coordinate_median([np.ones(3), np.ones(5)])


# ===========================================================================
# 3. Krum
# ===========================================================================


class TestKrum(unittest.TestCase):
    def _setup_krum(self, f: int = 1):
        """n=5 honest + 1 Byzantine. f=1. Krum should pick an honest update."""
        d = 20
        self.honest = _honest(5, d=d, center=0.0, noise=0.01)
        self.byzantine = np.full(d, 500.0, dtype=np.float64)
        self.all_updates = self.honest + [self.byzantine]
        self.f = f

    def test_krum_selects_honest_update(self):
        """Krum must not select the Byzantine update."""
        self._setup_krum()
        result = krum(self.all_updates, f=self.f)
        # Honest updates are all near 0; Byzantine is at 500
        self.assertLess(np.abs(result).max(), 1.0)

    def test_krum_returns_correct_shape(self):
        self._setup_krum()
        result = krum(self.all_updates, f=self.f)
        self.assertEqual(result.shape, self.all_updates[0].shape)

    def test_krum_with_no_byzantine(self):
        """With f=0, Krum picks the update closest to the geometric median."""
        n, d = 5, 10
        updates = _honest(n, d=d, center=1.0, noise=0.001)
        result = krum(updates, f=0)
        # Should be one of the updates — close to 1.0
        np.testing.assert_allclose(result, np.ones(d), atol=0.02)

    def test_invalid_f_raises(self):
        """2*f + 2 >= n must raise ValueError."""
        updates = [np.ones(5) for _ in range(4)]
        with self.assertRaises(ValueError):
            krum(updates, f=2)  # 2*2+2=6 >= 4

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            krum([], f=0)

    def test_mismatched_shapes_raises(self):
        with self.assertRaises(ValueError):
            krum([np.ones(3), np.ones(5)], f=0)


# ===========================================================================
# 4. Multi-Krum
# ===========================================================================


class TestMultiKrum(unittest.TestCase):
    def _setup(self, f: int = 1):
        d = 30
        self.honest = _honest(6, d=d, center=2.0, noise=0.01)
        self.byzantine = np.full(d, 5000.0, dtype=np.float64)
        self.all_updates = self.honest + [self.byzantine]
        self.f = f
        self.d = d

    def test_multi_krum_m1_same_as_krum(self):
        """With m=1, Multi-Krum must select exactly the same update as Krum."""
        self._setup()
        krum_result = krum(self.all_updates, f=self.f)
        mkrum_result = multi_krum(self.all_updates, f=self.f, m=1)
        np.testing.assert_array_equal(krum_result, mkrum_result)

    def test_multi_krum_excludes_byzantine(self):
        """Byzantine update (value 5000) must not influence the result."""
        self._setup()
        result = multi_krum(self.all_updates, f=self.f, m=3)
        # Honest centre ≈ 2.0; if Byzantine was included, result >> 2.0
        np.testing.assert_allclose(result, np.full(self.d, 2.0), atol=0.1)

    def test_multi_krum_default_m(self):
        """Default m = n - f should work without passing m."""
        self._setup()
        result = multi_krum(self.all_updates, f=self.f)
        self.assertEqual(result.shape, (self.d,))
        # Must still be near honest centre
        np.testing.assert_allclose(result, np.full(self.d, 2.0), atol=0.2)

    def test_multi_krum_all_honest(self):
        """With f=0 and m=n, result should equal the plain mean."""
        n, d = 5, 10
        updates = [np.full(d, float(i)) for i in range(n)]
        result = multi_krum(updates, f=0, m=n)
        expected = np.mean(updates, axis=0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_multi_krum_returns_correct_shape(self):
        self._setup()
        result = multi_krum(self.all_updates, f=self.f, m=2)
        self.assertEqual(result.shape, (self.d,))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            multi_krum([], f=0)


# ===========================================================================
# Cross-algorithm comparison
# ===========================================================================


class TestCrossAlgorithmComparison(unittest.TestCase):
    """All four algorithms should agree in the absence of Byzantine clients."""

    def test_all_agree_on_honest_data(self):
        """With no Byzantine clients, all four methods should produce similar results."""
        updates = _honest(10, d=20, center=5.0, noise=0.05)
        tm = trimmed_mean(updates, trim_ratio=0.1)
        cm = coordinate_median(updates)
        kr = krum(updates, f=0)
        mk = multi_krum(updates, f=0, m=8)

        target = np.full(20, 5.0)
        for name, result in [("trimmed_mean", tm), ("coord_median", cm),
                              ("krum", kr), ("multi_krum", mk)]:
            np.testing.assert_allclose(
                result, target, atol=0.3,
                err_msg=f"{name} diverged too far from honest centre"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
