"""
security.defenses
=================
Byzantine-robust aggregation algorithms for FedMed.

All functions accept a list of 1-D NumPy arrays (flattened model updates)
and return a single 1-D array representing the aggregated result.  The
calling code (Flower strategy or Go bridge) is responsible for reshaping
the output back to the original model parameter shapes.

Algorithms implemented
----------------------
1. **Trimmed Mean** — removes the top/bottom *trim_ratio* fraction of
   values per coordinate before averaging.  Tolerates up to
   ``floor(n * trim_ratio)`` Byzantine clients per coordinate.

2. **Coordinate-wise Median** — takes the element-wise median across all
   client updates.  Breakdown point ≈ 50 % — the most robust option against
   large-magnitude attacks, at the cost of some statistical efficiency.

3. **Krum** — selects the single update whose sum of squared distances to
   its ``n - f - 2`` nearest neighbours is minimal.  Breakdown point = f/n.
   ``f`` = known (or estimated) number of Byzantine clients.

4. **Multi-Krum** — computes Krum scores for all updates and returns the
   average of the *m* updates with the lowest scores.  Better statistical
   efficiency than Krum when ``m > 1``.

References
----------
* Blanchard et al. (2017) "Machine Learning with Adversaries: Byzantine
  Tolerant Gradient Descent."
* Yin et al. (2018) "Byzantine-Robust Distributed Learning: Towards Optimal
  Statistical Rates."
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Update = np.ndarray  # Expected shape: (d,) — flattened parameter vector


# ---------------------------------------------------------------------------
# 1. Trimmed Mean
# ---------------------------------------------------------------------------


def trimmed_mean(updates: List[Update], trim_ratio: float = 0.1) -> Update:
    """Coordinate-wise trimmed mean.

    For each dimension *d*, sorts the *n* values, discards the bottom and top
    ``floor(n * trim_ratio)`` entries, then averages the remainder.

    Parameters
    ----------
    updates:
        List of *n* 1-D arrays, all of the same length *d*.
    trim_ratio:
        Fraction of updates to remove from each tail.  Must be in
        ``[0, 0.5)``.  A value of 0.1 removes 10 % from each end.

    Returns
    -------
    np.ndarray
        Trimmed mean of shape ``(d,)``.

    Raises
    ------
    ValueError
        If *trim_ratio* is out of range or the remaining count is zero.
    """
    if not 0.0 <= trim_ratio < 0.5:
        raise ValueError(f"trim_ratio must be in [0, 0.5); got {trim_ratio}")
    if not updates:
        raise ValueError("updates must be non-empty")

    _assert_same_shape(updates)
    n = len(updates)
    k = int(np.floor(n * trim_ratio))

    if 2 * k >= n:
        raise ValueError(
            f"trim_ratio={trim_ratio} removes all updates ({2*k} ≥ {n}). "
            "Reduce trim_ratio or supply more clients."
        )

    stacked = np.stack(updates, axis=0)  # (n, d)
    stacked.sort(axis=0)
    trimmed = stacked[k : n - k, :]     # (n-2k, d)
    result = trimmed.mean(axis=0)

    logger.debug(
        "[defenses] trimmed_mean: n=%d, k=%d (removed %d per tail), "
        "result norm=%.4f",
        n, k, k, float(np.linalg.norm(result)),
    )
    return result


# ---------------------------------------------------------------------------
# 2. Coordinate-wise Median
# ---------------------------------------------------------------------------


def coordinate_median(updates: List[Update]) -> Update:
    """Element-wise median across all client updates.

    Parameters
    ----------
    updates:
        List of *n* 1-D arrays, all of the same length *d*.

    Returns
    -------
    np.ndarray
        Element-wise median of shape ``(d,)``.
    """
    if not updates:
        raise ValueError("updates must be non-empty")
    _assert_same_shape(updates)

    stacked = np.stack(updates, axis=0)  # (n, d)
    result = np.median(stacked, axis=0)

    logger.debug(
        "[defenses] coordinate_median: n=%d, result norm=%.4f",
        len(updates), float(np.linalg.norm(result)),
    )
    return result


# ---------------------------------------------------------------------------
# 3. Krum
# ---------------------------------------------------------------------------


def krum(updates: List[Update], f: int) -> Update:
    """Select the single update with the lowest Krum score.

    The Krum score of update *i* is the sum of its squared Euclidean
    distances to its ``n - f - 2`` nearest neighbours (excluding itself).

    Parameters
    ----------
    updates:
        List of *n* 1-D arrays, all of the same length *d*.
    f:
        Number of Byzantine (potentially malicious) clients to tolerate.
        Must satisfy ``2*f + 2 < n``.

    Returns
    -------
    np.ndarray
        The selected update of shape ``(d,)``.
    """
    scores, selected_idx = _krum_scores(updates, f)
    selected = updates[selected_idx]

    logger.debug(
        "[defenses] krum: n=%d, f=%d, selected client %d "
        "(score=%.4f), result norm=%.4f",
        len(updates), f, selected_idx, scores[selected_idx],
        float(np.linalg.norm(selected)),
    )
    return selected


# ---------------------------------------------------------------------------
# 4. Multi-Krum
# ---------------------------------------------------------------------------


def multi_krum(updates: List[Update], f: int, m: Optional[int] = None) -> Update:
    """Average the *m* updates with the lowest Krum scores.

    Parameters
    ----------
    updates:
        List of *n* 1-D arrays, all of the same length *d*.
    f:
        Number of Byzantine clients to tolerate.
    m:
        Number of updates to select and average.  Defaults to ``n - f``.

    Returns
    -------
    np.ndarray
        Mean of the *m* selected updates, shape ``(d,)``.
    """
    n = len(updates)
    if m is None:
        m = n - f
    m = max(1, min(m, n))

    scores, _ = _krum_scores(updates, f)
    top_m_indices = np.argsort(scores)[:m]
    selected = [updates[i] for i in top_m_indices]
    result = np.mean(np.stack(selected, axis=0), axis=0)

    logger.debug(
        "[defenses] multi_krum: n=%d, f=%d, m=%d, "
        "selected indices=%s, result norm=%.4f",
        n, f, m, list(top_m_indices), float(np.linalg.norm(result)),
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _krum_scores(
    updates: List[Update], f: int
) -> Tuple[np.ndarray, int]:
    """Compute Krum scores for all updates and return (scores, argmin).

    Parameters
    ----------
    updates:
        List of *n* updates.
    f:
        Byzantine tolerance.  Requires ``2*f + 2 < n``.
    """
    if not updates:
        raise ValueError("updates must be non-empty")
    _assert_same_shape(updates)

    n = len(updates)
    if 2 * f + 2 >= n:
        raise ValueError(
            f"Krum requires 2*f + 2 < n, but 2*{f} + 2 = {2*f+2} ≥ {n}. "
            "Reduce f or add more clients."
        )

    stacked = np.stack(updates, axis=0)  # (n, d)
    k = n - f - 2  # number of neighbours to consider

    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # Squared Euclidean distances from update i to all others
        diffs = stacked - stacked[i]           # (n, d)
        dist_sq = (diffs ** 2).sum(axis=1)     # (n,)
        dist_sq[i] = np.inf                    # exclude self
        nearest_k = np.partition(dist_sq, k - 1)[:k]
        scores[i] = nearest_k.sum()

    return scores, int(np.argmin(scores))


def _assert_same_shape(updates: List[Update]) -> None:
    """Raise ValueError if updates have inconsistent shapes."""
    shapes = {u.shape for u in updates}
    if len(shapes) > 1:
        raise ValueError(f"All updates must have the same shape; got {shapes}")
