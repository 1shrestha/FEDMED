"""Generate the SYNTHETIC per-hospital datasets into hospital_nodes/<id>/local_data.npz."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import get_settings
from backend.utils.data_gen import generate_all

if __name__ == "__main__":
    generate_all(get_settings().data_dir)
    print(f"Synthetic hospital datasets written to {get_settings().data_dir}")
