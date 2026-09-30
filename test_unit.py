import numpy as np
import pytest
from backend.evaluation.metrics import confusion_matrix, scores_from_cm
from backend.federated.fedavg import fedavg
from backend.privacy.privacy import assert_no_raw_data, clip_update
from backend.training.model import init_params, local_train, log_loss
from backend.utils.data_gen import HOSPITAL_PROFILES, N_FEATURES, make_hospital_data


def test_fedavg_is_sample_weighted_mean():
    out = fedavg([np.array([1.0, 1.0]), np.array([4.0, 7.0])], [1, 3])
    assert np.allclose(out, [3.25, 5.5])


def test_fedavg_rejects_bad_input():
    with pytest.raises(ValueError):
        fedavg([np.zeros(2), np.zeros(3)], [1, 1])
    with pytest.raises(ValueError):
        fedavg([], [])


def test_model_initialization_shape_and_small_values():
    theta = init_params(N_FEATURES, seed=1)
    assert theta.shape == (N_FEATURES + 1,) and np.abs(theta).max() < 0.1


def test_local_training_reduces_loss():
    d = make_hospital_data(**HOSPITAL_PROFILES["hospital_2"], rng=np.random.default_rng(0))
    t0 = init_params(N_FEATURES)
    t1 = local_train(t0, d["X_train"], d["y_train"], epochs=3, lr=0.1)
    assert log_loss(t1, d["X_train"], d["y_train"]) < log_loss(t0, d["X_train"], d["y_train"])


def test_local_training_empty_dataset_fails():
    with pytest.raises(ValueError):
        local_train(init_params(3), np.empty((0, 3)), np.empty(0), 1, 0.1)


def test_metrics_known_values():
    cm = confusion_matrix([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert cm == [[1, 1], [1, 2]]
    s = scores_from_cm(cm)
    assert s["accuracy"] == pytest.approx(0.6) and s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(2 / 3) and s["f1"] == pytest.approx(2 / 3)


def test_partitions_are_non_iid():
    rates = {}
    for hid, p in HOSPITAL_PROFILES.items():
        d = make_hospital_data(**p, rng=np.random.default_rng(0))
        rates[hid] = d["y_train"].mean()
    assert rates["hospital_1"] < 0.3 < 0.55 < rates["hospital_2"]


def test_clipping_bounds_norm():
    assert np.linalg.norm(clip_update(np.ones(9) * 10, 1.0)) == pytest.approx(1.0)


def test_privacy_validation_rejects_records():
    ok = dict(hospital_id="h", round=1, num_samples=5, delta=[0.1] * (N_FEATURES + 1))
    assert_no_raw_data(ok, N_FEATURES)
    with pytest.raises(ValueError):
        assert_no_raw_data({**ok, "patient_records": [[1.0, 2.0]]}, N_FEATURES)
    with pytest.raises(ValueError):
        assert_no_raw_data({**ok, "delta": [[1.0] * 3] * 9}, N_FEATURES)
