"""Completion rate prediction service.

Environment variables:
- MODEL_DIR: Path to the directory where models are saved. Default `./model`.
- MAX_NUM_MODELS: Maxium number of models sustained at the same time. Default 1.
"""

import os
from pathlib import Path
from collections import OrderedDict
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb

from preprocess import Processor

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "model"))


class LRU(OrderedDict):
    """Container to save models in memory.

    Limit size, removing the oldest model when full.
    """

    def __init__(self, *args, **kwds) -> None:
        self.maxsize = os.environ.get("MAX_NUM_MODELS", 1)
        super().__init__(*args, **kwds)

    def __setitem__(self, key, value) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

    def latest(self):
        return self[next(reversed(self))]


class Record(BaseModel):
    """Features of a form."""
    feat_01: int
    feat_02: int
    feat_03: int
    feat_04: int
    feat_05: int
    feat_06: int
    feat_07: int
    feat_08: int
    feat_09: int
    feat_10: int
    feat_11: int
    feat_12: int
    feat_13: int
    feat_14: int
    feat_15: int
    feat_16: int
    feat_17: int
    feat_18: int
    feat_19: int
    feat_20: int
    feat_21: int
    feat_22: int
    feat_23: int
    feat_24: int
    feat_25: int
    feat_26: int
    feat_27: int
    feat_28: int
    feat_29: int
    feat_30: int
    feat_31: int
    feat_32: int
    feat_33: int
    feat_34: int
    feat_35: int
    feat_36: int
    feat_37: int
    feat_38: int
    feat_39: int
    feat_40: int
    feat_41: int
    feat_42: int
    feat_43: int
    feat_44: int
    feat_45: int
    feat_46: int
    feat_47: int


class ModelID(BaseModel):
    """ID of the model."""
    model_id: int


class Score(BaseModel):
    """Probablity of submition."""
    score: float


app = FastAPI()
Models = LRU()


def _load_model(model_id: int = None) -> None:
    """Load a model and processor to memory.

    Parameters
    ----------
    model_id :
        ID of the model to load. If omit, load
        the model with the highest name in model
        directory. (model ID is epoch timestamp.)
    """
    if not model_id:
        model_file = sorted(MODEL_DIR.glob("*.model")).pop()
    else:
        model_file = MODEL_DIR / f"{model_id}.model"
        if not model_file.exists():
            raise FileNotFoundError(model_file)

    bst = xgb.Booster()
    bst.load_model(model_file)
    model = {"model": bst}
    processor = model_file.with_suffix('.processor')
    if processor.exists():
        with processor.open('rb') as inf:
            model['processor'] = pickle.load(inf)

    Models[model_id] = model


def infer(model: dict, record: Record) -> float:
    """Infer the complition rate.

    Parameters
    ----------
    model :
        the model and processor used to infer the
        compeletion rate.
    record :
        the form features to infer the completion
        rate

    Returns
    -------
    float:
        the percentage of users to submit this form
        after viewing.
    """
    df = pd.DataFrame(dict(record), index=[0])
    if processor := model.get('processor'):
        df = processor.pipe(df, infer=True)
    data = xgb.DMatrix(df)
    result = model["model"].predict(data)
    return result[0]


_load_model()


@app.get("/")
async def heartbeat():
    """health check."""
    return "ok"


@app.post("/infer/", response_model=Score)
def infer_with_latest_model(record: Record):
    """Preidct completion rate with the default model."""
    model = Models.latest()
    score = infer(model, record)
    return {'score': score}


@app.post("/infer/{model_id}", response_model=Score)
def infer_with_model_id(model_id: int, record: Record):
    """Predict completion rate with a specific model."""
    if model_id not in Models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    model = Models[model_id]
    score = infer(model, record)
    return {'score': score}


@app.post("/load/{model_id}", response_model=ModelID)
def load_model(model_id: int):
    """Load a prediction model to memory and use it as defult."""
    try:
        _load_model(model_id)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404, detail=f"Model {model_id} not found"
        ) from e
    return {"model_id": model_id}
