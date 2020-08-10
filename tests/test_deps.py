from pathlib import Path

import xgboost as xgb
from pytest import fixture

from app.api.deps import LRUModelContainer

MODEL_DIR = "tests/test_models"


@fixture
def models():
    return LRUModelContainer(model_dir=MODEL_DIR, max_size=2)


def test_latest(models):
    models["0"] = 0
    models["1"] = 1
    assert models.latest() == 1


def test_find_latest_version(mocker, models):
    mocked_glob = mocker.patch("pathlib.Path.glob")
    mocked_glob.return_value = [Path("/0.model"), Path("/1.model")]

    assert models._find_latest_version() == 1


def test_read_model_file(mocker, models):
    mocked_exists = mocker.patch("pathlib.Path.exists")
    mocked_exists.return_value = True

    assert str(models._read_model_file(version=42)).endswith(f"{MODEL_DIR}/42.model")


def test_unpickle_model_file(models):
    model_file = Path(f"{MODEL_DIR}/1596729586.model").resolve()
    model = models._unpickle_model_file(model_file=model_file)

    assert isinstance(model["model"], xgb.Booster)


def test_insert_model_order(models):
    models._insert_model(1, "")
    assert list(models.keys()) == [1]
    models._insert_model(2, "")
    assert list(models.keys()) == [1, 2]
    models._insert_model(1, "")
    assert list(models.keys()) == [2, 1]
    models._insert_model(3, "")
    assert list(models.keys()) == [1, 3]
