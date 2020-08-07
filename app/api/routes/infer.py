"""Routes for predicting form completion rate."""

from fastapi import APIRouter, HTTPException

from app.api import Models
from app.api.deps import infer
from app.api.model import Record, Score, Versions

router = APIRouter()


@router.get("/", response_model=Versions)
async def list_available_model_ids():
    """Preidct completion rate with the default model."""
    return {"model_ids": list(Models.keys())}


@router.post("/", response_model=Score)
async def infer_with_latest_model(record: Record):
    """Preidct completion rate with the default model."""
    model = Models.latest()
    score = infer(model, record)
    return {"score": score}


@router.post("/{version}", response_model=Score)
async def infer_with_model_id(version: int, record: Record):
    """Predict completion rate with a specific model."""
    if version not in Models.keys():
        raise HTTPException(status_code=404, detail=f"Model {version} not found")
    model = Models[version]
    score = infer(model, record)
    return {"score": score}
