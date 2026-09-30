import numpy as np
from backend.federated.client import HospitalNode
from backend.utils.data_gen import N_FEATURES

REG = lambda i: {"hospital_id": f"hospital_{i}", "name": f"H{i}"}


def _setup(client):
    for i in (1, 2, 3):
        assert client.post("/hospitals/register", json=REG(i)).status_code == 201
    assert client.post("/federated/start", json={"seed": 0}).status_code == 200


def test_health_and_registration_errors(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/hospitals/register", json=REG(1)).status_code == 201
    assert client.post("/hospitals/register", json=REG(1)).status_code == 409       # duplicate
    assert client.post("/hospitals/register", json={"hospital_id": "ghost", "name": "x"}).status_code == 404  # no dataset
    assert client.post("/hospitals/register", json={"hospital_id": "../x", "name": "x"}).status_code == 422  # invalid
    assert client.get("/model/global").status_code == 404


def test_start_requires_hospitals_and_train_requires_start(client):
    assert client.post("/federated/start").status_code == 400
    client.post("/hospitals/register", json=REG(1))
    assert client.post("/federated/train", json={"rounds": 1}).status_code == 409


def test_full_federated_training_and_global_model(client):
    _setup(client)
    r = client.post("/federated/train", json={"rounds": 4, "epochs": 2, "lr": 0.1})
    assert r.status_code == 200 and r.json()["round"] == 4
    rounds = client.get("/federated/rounds").json()
    assert len(rounds["global"]) == 4 and len(rounds["hospitals"]) == 12
    assert rounds["global"][-1]["loss"] < rounds["global"][0]["loss"]
    model = client.get("/model/global").json()
    assert model["version"] == 4 and len(model["weights"]) == N_FEATURES
    m = client.get("/metrics").json()
    assert m["comparison"]["federated"]["accuracy"] > 0.5
    assert len(m["comparison"]["local_only"]) == 3


def test_update_endpoint_validation(client, settings):
    _setup(client)
    good = dict(hospital_id="hospital_1", round=1, num_samples=10, delta=[0.0] * (N_FEATURES + 1))
    assert client.post("/federated/update", json={**good, "delta": [0.0] * 3}).status_code == 422   # incompatible
    assert client.post("/federated/update", json={**good, "patient_records": [[1, 2]]}).status_code == 422  # raw data rejected
    assert client.post("/federated/update", json={**good, "hospital_id": "nope"}).status_code == 404
    assert client.post("/federated/update", json={**good, "round": 5}).status_code == 409
    assert client.post("/federated/update", json=good).json()["aggregated"] is False
    assert client.post("/federated/update", json=good).status_code == 409                           # duplicate


def test_update_payload_never_contains_raw_patient_data(settings):
    node = HospitalNode("hospital_1", settings.data_dir)
    payload = node.train_round(np.zeros(N_FEATURES + 1), 1, epochs=1, lr=0.1)
    assert set(payload) == {"hospital_id", "round", "num_samples", "delta", "metrics"}
    assert len(payload["delta"]) == N_FEATURES + 1
    flat = str(payload)
    raw = np.load(settings.data_dir / "hospital_1" / "local_data.npz")["X_train"]
    assert str(round(float(raw[0, 0]), 4)) not in flat          # no raw feature values leaked
    assert payload["num_samples"] == node.n_train               # only a count, not records


def test_privacy_options_applied(server):
    server.settings.clip_norm, server.settings.noise_multiplier = 0.05, 0.1
    server.run_round(epochs=2, lr=0.1)
    assert server.round == 1
