import pickle
from pathlib import Path
from unittest.mock import mock_open

import numpy as np
import pytest
import xgboost as xgb
from freezegun import freeze_time

from app.estimator import train


@pytest.fixture()
def data():
    return xgb.DMatrix(np.random.rand(5, 10), label=np.random.randint(2, size=5))


def test_load_not_exist_data():
    with pytest.raises(FileNotFoundError):
        train.load_data("foo")


def test_cv(mocker, data):
    spy_xgb = mocker.spy(xgb, "cv")
    num_boost_round = 1
    nfold = 2
    params = {"eta": 1, "objective": "reg:squarederror", "eval_metric": "rmse"}
    other_params = {"max_depth": 2.2}

    loss = train.cv(
        data=data,
        num_boost_round=num_boost_round,
        nfold=nfold,
        params=params,
        **other_params
    )

    spy_xgb.assert_called_once_with(
        {
            "max_depth": 2,
            "eta": 1,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
        },
        data,
        num_boost_round=num_boost_round,
        nfold=nfold,
    )
    assert isinstance(loss, float)


def test_train(mocker, data):
    mocked_xgb = mocker.patch.object(xgb, "train")
    params = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"}
    num_boost_round = 1
    train.train(data, params=params, num_boost_round=num_boost_round)

    mocked_xgb.assert_called_once_with(params, data, num_boost_round=num_boost_round)


@freeze_time("1970-01-01 00:00:01")
def test_export_filename(mocker):
    sample_model = xgb.Booster()
    sample_processer = mocker.stub(name="sample_processor")

    mocked_save_model = mocker.patch.object(sample_model, "save_model")
    mocker.patch.object(pickle, "dump")
    opener = mock_open()

    def mocked_open(self, *args, **kwargs):
        return opener(self, *args, **kwargs)

    mocker.patch.object(Path, "open", mocked_open)

    train.export(sample_model, sample_processer)

    mocked_save_model.assert_called_once_with("1.model")
    opener.assert_called_once_with(Path("1.processor"), "wb")
