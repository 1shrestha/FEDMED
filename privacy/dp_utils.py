"""
dp_utils.py  (WEEK 4 DELIVERABLE)
-----------------------------------
Adds calibrated Gaussian noise to weight updates before transmission, to
protect against model-inversion / membership-inference attacks (an
attacker with access to the aggregated model trying to reconstruct or
infer facts about individual patients' training data).

This implements the Gaussian mechanism for (epsilon, delta)-differential
privacy at the level of each client's weight UPDATE (the delta between the
weights the client received and the weights after local training) — this
is the standard "DP-FedAvg" pattern and is much simpler to defend in a
review than full DP-SGD (per-sample gradient clipping every training step).

For your report: be explicit that epsilon is a *privacy budget* — smaller
epsilon = more noise = stronger privacy but worse accuracy. Show a small
sweep (e.g. epsilon = 1, 5, 10) and the resulting Dice score, so the
review sees you understand the tradeoff rather than picking one number
arbitrarily.
"""

import numpy as np


def clip_update(update: np.ndarray, clip_norm: float = 1.0) -> np.ndarray:
    """Clips the L2 norm of a weight update — required before adding calibrated noise,
    otherwise a single outlier update could blow the privacy guarantee."""
    norm = np.linalg.norm(update)
    if norm > clip_norm:
        update = update * (clip_norm / norm)
    return update


def gaussian_noise_scale(clip_norm: float, epsilon: float, delta: float = 1e-5) -> float:
    """
    Standard analytic Gaussian mechanism noise scale (sigma) for a given
    privacy budget (epsilon, delta). Smaller epsilon -> larger sigma -> more noise.
    """
    return (clip_norm / epsilon) * np.sqrt(2 * np.log(1.25 / delta))


def add_dp_noise(update: np.ndarray, epsilon: float = 5.0, delta: float = 1e-5, clip_norm: float = 1.0) -> np.ndarray:
    """
    Full DP-FedAvg client-side step:
      1. Clip the update's L2 norm (bounds sensitivity).
      2. Add Gaussian noise calibrated to (epsilon, delta).

    Call this on the CLIENT before sending the update to the server
    (before or after HE encryption — noise should be added in plaintext,
    pre-encryption, since it needs to affect the actual values).
    """
    clipped = clip_update(update, clip_norm)
    sigma = gaussian_noise_scale(clip_norm, epsilon, delta)
    noise = np.random.normal(loc=0.0, scale=sigma, size=clipped.shape)
    return clipped + noise


if __name__ == "__main__":
    fake_update = np.random.normal(0, 0.5, size=1000)
    for eps in [1, 5, 10]:
        noisy = add_dp_noise(fake_update, epsilon=eps)
        print(f"epsilon={eps:>2}  noise_std~{gaussian_noise_scale(1.0, eps):.3f}  "
              f"mean_abs_diff_from_original={np.mean(np.abs(noisy - fake_update)):.4f}")
