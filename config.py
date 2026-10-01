"""Configuration loaded from environment / .env (no secrets are hardcoded)."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


@dataclass
class Settings:
    data_dir: Path                      # where each hospital's LOCAL data lives
    db_path: Path                       # results database (metrics only, no patient data)
    clip_norm: Optional[float] = None   # update clipping (privacy, optional)
    noise_multiplier: float = 0.0       # Gaussian noise on updates (optional, no formal epsilon)


def get_settings() -> Settings:
    _load_dotenv()
    clip = os.getenv("FEDMED_CLIP_NORM", "").strip()
    return Settings(
        data_dir=Path(os.getenv("FEDMED_DATA_DIR", ROOT / "hospital_nodes")),
        db_path=Path(os.getenv("FEDMED_DB_PATH", ROOT / "fedmed_results.db")),
        clip_norm=float(clip) if clip else None,
        noise_multiplier=float(os.getenv("FEDMED_NOISE_MULTIPLIER", "0") or 0),
    )
