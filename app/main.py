"""FedMed Flower application entry point.

All application dependency construction is centralized in
``src.fl.orchestrator``.  Flower loads these two exported objects from this
module according to ``pyproject.toml``.
"""

from src.fl.orchestrator import FedMedOrchestrator


orchestrator = FedMedOrchestrator()
client_app, server_app = orchestrator.build_apps()


__all__ = ["client_app", "server_app"]
