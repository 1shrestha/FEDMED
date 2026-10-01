from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class HospitalRegistration(BaseModel):
    # restricted charset also prevents path traversal into the data directory
    hospital_id: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,32}$")
    name: str = Field(min_length=1, max_length=100)


class UpdateMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    local_accuracy: float
    local_loss: float


class ModelUpdate(BaseModel):
    """The ONLY thing a hospital may send: a parameter delta. Extra fields are rejected,
    so raw patient records cannot be smuggled in through the API."""
    model_config = ConfigDict(extra="forbid")
    hospital_id: str
    round: int = Field(ge=1)
    num_samples: int = Field(gt=0)
    delta: list[float]
    metrics: Optional[UpdateMetrics] = None


class StartRequest(BaseModel):
    seed: int = 0


class TrainRequest(BaseModel):
    rounds: int = Field(5, ge=1, le=100)
    epochs: int = Field(3, ge=1, le=50)
    lr: float = Field(0.1, gt=0, le=5)
