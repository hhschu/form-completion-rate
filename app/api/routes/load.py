"""Routes for hot-loading prediction models."""

from fastapi import APIRouter, HTTPException

from app.api import Models
from app.api.model import Version

router = APIRouter()


@router.post("/{model_id}", response_model=Version)
async def load_model(version: int):
    """Load a prediction model to memory and use it as defult."""
    try:
        Models.load(version)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Model {version} not found") from e
    return {"version": version}
