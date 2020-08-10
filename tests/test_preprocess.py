import numpy as np
import pandas as pd
from pytest import fixture
from sklearn.preprocessing import StandardScaler

from app.estimator import preprocess


@fixture
def processor():
    return preprocess.Processor()


@fixture
def data():
    return pd.DataFrame(
        {
            "views": [100, 1000],
            "submissions": [90, 100],
            "feat_01": [0, 0],
            "feat_02": [1, 0],
        }
    )


def test_select_features(processor, data):
    processor.select_features(data)

    assert processor.features == ["feat_02"]


def test_training_pipe(mocker, processor, data):
    mocked_wilson_lower_bound = mocker.spy(preprocess, "wilson_lower_bound")
    mocked_select_feature = mocker.patch.object(preprocess.Processor, "select_features")
    mocked_fit = mocker.patch.object(StandardScaler, "fit")
    mocked_transform = mocker.patch.object(StandardScaler, "transform")

    processor.pipe(data, infer=False)

    mocked_wilson_lower_bound.assert_called_once()
    mocked_select_feature.assert_called_once()
    mocked_fit.assert_called_once()
    mocked_transform.assert_called_once()


def test_infer_pipe(mocker, processor, data):
    mocked_wilson_lower_bound = mocker.spy(preprocess, "wilson_lower_bound")
    mocked_select_feature = mocker.patch.object(preprocess.Processor, "select_features")
    mocked_fit = mocker.patch.object(StandardScaler, "fit")
    mocked_transform = mocker.patch.object(StandardScaler, "transform")

    processor.pipe(data, infer=True)

    mocked_wilson_lower_bound.assert_not_called()
    mocked_select_feature.assert_not_called()
    mocked_fit.assert_not_called()
    mocked_transform.assert_called_once()


def test_wilson_lower_bound(data):
    actual = preprocess.wilson_lower_bound(
        submissions=data["submissions"], views=data["views"]
    )
    expected = np.array([0.825633, 0.082909])

    np.testing.assert_allclose(actual, expected, atol=1e-6)
