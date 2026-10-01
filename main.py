"""FedMed API. Run: uvicorn backend.main:app --reload"""
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from backend.config import Settings, get_settings
from backend.database.results import ResultsStore
from backend.errors import FedMedError
from backend.federated.server import FederatedServer
from backend.schemas import HospitalRegistration, ModelUpdate, StartRequest, TrainRequest

INDEX_HTML = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    server = FederatedServer(settings, ResultsStore(settings.db_path))
    app = FastAPI(title="FedMed", description="Federated learning SIMULATION for medical AI")

    @app.exception_handler(FedMedError)
    async def _fedmed_error(_: Request, exc: FedMedError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return INDEX_HTML.read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/hospitals/register", status_code=201)
    def register(body: HospitalRegistration):
        return server.register(body.hospital_id, body.name)

    @app.get("/hospitals")
    def hospitals():
        return list(server.hospitals.values())

    @app.post("/federated/start")
    def start(body: StartRequest = StartRequest()):
        return server.start(body.seed)

    @app.post("/federated/train")
    def train(body: TrainRequest = TrainRequest()):
        for _ in range(body.rounds):
            server.run_round(body.epochs, body.lr)
        server.run_baseline(server.round * body.epochs, body.lr)
        return {**server.status_dict(), "global": server.store.global_rounds()[-1]}

    @app.post("/federated/update")
    def update(body: ModelUpdate):
        return {"aggregated": server.submit_update(body), **server.status_dict()}

    @app.get("/federated/status")
    def status():
        return server.status_dict()

    @app.get("/federated/rounds")
    def rounds():
        return {"global": server.store.global_rounds(), "hospitals": server.store.hospital_rounds()}

    @app.get("/model/global")
    def global_model():
        if server.theta is None:
            raise FedMedError(404, "No global model yet; call POST /federated/start")
        return {"version": server.version, "round": server.round,
                "weights": server.theta[:-1].tolist(), "bias": float(server.theta[-1])}

    @app.get("/metrics")
    def metrics():
        g = server.store.global_rounds()
        return {"latest_global": g[-1] if g else None, "comparison": server.store.get_kv("comparison")}

    return app


app = create_app()
