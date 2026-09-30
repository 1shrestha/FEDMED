import pytest
from fastapi.testclient import TestClient
from backend.config import Settings
from backend.database.results import ResultsStore
from backend.federated.server import FederatedServer
from backend.main import create_app
from backend.utils.data_gen import generate_all


@pytest.fixture()
def settings(tmp_path):
    generate_all(tmp_path / "nodes")
    return Settings(data_dir=tmp_path / "nodes", db_path=tmp_path / "r.db")


@pytest.fixture()
def server(settings):
    srv = FederatedServer(settings, ResultsStore(settings.db_path))
    for i in (1, 2, 3):
        srv.register(f"hospital_{i}", f"H{i}")
    srv.start()
    return srv


@pytest.fixture()
def client(settings):
    return TestClient(create_app(settings))
