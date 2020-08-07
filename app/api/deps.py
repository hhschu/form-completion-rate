"""Utility functions."""

import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb

from app.api.model import Record

logger = logging.getLogger("serving")


class PreprocessUnpickler(pickle.Unpickler):
    """Unpickle a preporcess object.

    Handles the difference in project structure
    between training and serving.
    """

    def find_class(self, module: str, name: Any):
        renamed_module = module
        if module == "preprocess":
            renamed_module = "app.estimator.preprocess"

        return super().find_class(renamed_module, name)


class LRUModelContainer(OrderedDict):
    """Container to save models in memory.

    The container has a limit size. When it's full,
    it removes the oldest loaded model.

    Parameters
    ----------
    model_dir :
        Path to the directory where models are stored.
    maxsize : (optional)
        maximum number of models. Removing the oldest
        model when full. Default 1.
    """

    def __init__(
        self, model_dir: str, maxsize: int = 1, *args: Any, **kwargs: Any
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.maxsize = maxsize
        super().__init__(*args, **kwargs)

    def latest(self) -> dict:
        """Return the lastet model."""
        return self[next(reversed(self))]

    def load(self, version: int = None) -> None:
        """Load a model and processor to memory.

        Parameters
        ----------
        version :
            Version number of the model to load. If omit,
            load the latest version. (version number is
            training epoch timestamp.)
        """
        if not version:
            version = self._find_latest_version()

        logger.info(f"loading model version {version}")
        model_file = self._read_model_file(version)
        model = self._unpickle_model_file(model_file)

        if version in self:
            self.move_to_end(version)

        super().__setitem__(version, model)

        if len(self) > self.maxsize:
            oldest = next(iter(self))
            logger.info(f"dropping model version {oldest}")
            del self[oldest]

    def _find_latest_version(self) -> int:
        model_file = sorted(self.model_dir.glob("*.model")).pop()
        if not model_file:
            raise FileNotFoundError(f"model folder {self.model_dir} is empty")
        return int(model_file.stem)

    def _read_model_file(self, version: int) -> Path:
        model_file = self.model_dir / f"{version}.model"
        if not model_file.exists():
            raise FileNotFoundError(model_file)
        return model_file

    def _unpickle_model_file(self, model_file: Path) -> dict:
        bst = xgb.Booster()
        bst.load_model(model_file)
        model = {"model": bst}
        processor = model_file.with_suffix(".processor")
        if processor.exists():
            with processor.open("rb") as inf:
                model["processor"] = PreprocessUnpickler(inf).load()
        return model


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
    if processor := model.get("processor"):
        df = processor.pipe(df, infer=True)
    data = xgb.DMatrix(df)
    result = model["model"].predict(data)
    return result[0]
