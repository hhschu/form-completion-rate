"""Run this file to train the prediction model."""
import argparse
import logging
import pickle
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import pandas as pd
import xgboost as xgb
from bayes_opt import BayesianOptimization
from preprocess import Processor  # type: ignore

logger = logging.getLogger("training")
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="Path to the training dataset.")
parser.add_argument("--steps", type=int, help="Number of boost round.")
parser.add_argument("--output", help="Path to the location to save the model.")
parser.add_argument("-v", "--verbose", action="store_true", help="Print more logs.")


def load_data(path: str) -> pd.DataFrame:
    """Load the training CSV dataset.

    Parameters
    ----------
    path :
        The path of the CSV data file.
    """
    start = time.time()
    p = Path(path).resolve()
    if not (p.is_file and p.exists()):
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    duration = time.time() - start
    logger.info(f"{len(df):,} rows loaded from {p} in {duration:.2f} seconds")
    return df


def cv(
    data: xgb.DMatrix,
    num_boost_round: int,
    nfold: int,
    params: Dict[str, Any] = None,
    **other_params: Any,
) -> float:
    """Run cross-validation on RMSE.

    Parameters
    ----------
    data :
        The training data.
    num_boost_round :
        Number of boost rounds for the XGBoost model.
    nfold :
        Number of fold to use in cross-validation.
    params :
        Model parameters that are not optimisable.
    **other_params:
        Other training parameters. This is intend
        to be provided in the optimise_hyper_params()
        function.
    """
    if "max_depth" in other_params:
        other_params["max_depth"] = int(other_params["max_depth"])

    params = {**(params or {}), **other_params}
    result = xgb.cv(params, data, num_boost_round=num_boost_round, nfold=nfold)
    return -1 * result[f"test-{params['eval_metric']}-mean"].iloc[-1]


def optimise_hyper_params(
    func: Callable[[dict], float],
    bounds: Dict[str, Tuple[float, float]],
    init_points: int = 3,
    n_iter: int = 5,
) -> Dict[str, float]:
    """Optimise hyper-parameters with Bayesian method.

    Parameters
    ----------
    fucn :
        The training function to optimise for. The function
        should take the parameters in `bounds` and return
        the metric to be maximise.
    bound :
        The hyper-parameters and their bounds to be optimised.
        It shoud be in the form of:
        ```
        {
            "param1": (lower, upper),
            "param2": (lower, upper)
        }
        ```
    init_points : (optional)
        Number of iteration to randomly search. Defualt 3.
    n_iter : (optional)
        Number to interation to search for the best parameters.
        Default 5.

    Returns
    -------
    dict :
        The best heper-parameters found.
    """
    start = time.time()
    optimizer = BayesianOptimization(f=func, pbounds=bounds)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    params = optimizer.res[-1]["params"]
    duration = time.time() - start
    logger.info(f"best hyper-parameter {params} found in {duration:.2f} seconds.")
    return params


def train(data: xgb.DMatrix, params: dict, num_boost_round: int) -> xgb.Booster:
    """Train a completion rate prediction model.

    Parameters
    ----------
    data :
        The training data, including both feature and ground truth.
    params :
        Hyper-parameters of the model. See https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
        for the full list.
    num_boost_round :
        Numer of boost round for the model.

    Returns
    -------
    xgb.Booster :
        The trained model.
    """
    start = time.time()
    model = xgb.train(params, data, num_boost_round=num_boost_round)
    duration = time.time() - start
    logger.info(f"model trained in {duration:.2f} seconds.")
    return model


def export(model: xgb.Booster, path: str, processer: Processor = None) -> None:
    """Export the prediction model.

    Parameters
    ----------
    model :
        The model to be exported.
    path :
        Path of the folder to save the model.
    processer : (optional)
        The accompany data preprocessor of the model.
    """
    ts = int(time.time())
    p = Path(path)
    logger.info(f"exporting model and processer to {p.resolve()}")
    model.save_model(str(p / f"{ts}.model"))
    if processer:
        with (p / f"{ts}.processor").open("wb") as outf:
            pickle.dump(processer, outf)


def main(flags: argparse.Namespace) -> None:
    raw_data = load_data(flags.data)
    processor = Processor()
    X, y = processor.pipe(raw_data, infer=False)
    data = xgb.DMatrix(X, label=y)

    params = {
        "eval_metric": "rmse",
        "objective": "reg:squarederror",
        "eta": 0.1,
    }
    func = partial(cv, data=data, num_boost_round=flags.steps, nfold=10, params=params)
    pbounds = {
        "max_depth": (2, 12),
        "subsample": (0.4, 1.0),
        "colsample_bytree": (0.4, 1.0),
    }
    optimised_params = optimise_hyper_params(func, pbounds)
    if "max_depth" in optimised_params:
        optimised_params["max_depth"] = int(optimised_params["max_depth"])

    model = train(
        data, params={**params, **optimised_params}, num_boost_round=flags.steps,
    )

    export(model=model, processer=processor, path=flags.output)


if __name__ == "__main__":
    flags = parser.parse_args()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG if flags.verbose else logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)-6s %(levelname)-8s %(message)s")
    )
    logger.addHandler(stream_handler)

    main(flags)
