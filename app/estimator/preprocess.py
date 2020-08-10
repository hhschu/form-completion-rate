"""Data preprocessing pipeline."""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Processor:
    """The data preprocessing pipeline."""

    def __init__(self) -> None:
        self.features: List[str] = []
        self.standardiser = StandardScaler()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def select_features(self, data: pd.DataFrame) -> None:
        """Select useful features."""
        s = (data[[c for c in data.columns if c.startswith("feat")]] == 0).mean()
        self.features = s[s < 0.9].index.to_list()

    def pipe(self, data: pd.DataFrame, infer: bool) -> Tuple:
        """The preprocessing pipeline.

        Parameters
        ----------
        data :
            Training or inferance data.
        infer :
            Model of the pipeline. If it's inferance.
        """
        y = None
        if not infer:
            data["completion_rate"] = wilson_lower_bound(
                data["submissions"], data["views"]
            )
            self.select_features(data)
            y = data.pop("completion_rate")
        X = data[self.features]

        if not infer:
            self.standardiser.fit(X)
        X = self.standardiser.transform(X)

        if not infer:
            assert y is not None
            return X, y
        return X


def wilson_lower_bound(
    submissions: np.array, views: np.array, z: float = 1.96
) -> np.array:
    """Calculate the lower confidance interval of Wilson score.

    Parameters
    ----------
    submissions :
        Number of submission, the numerator.
    views :
        Number of views, the denominator.
    z :
        Standard score. Default 1.96, 95% confidence interval.

    Returns
    -------
    np.array
        The Wilson score adjuscted completion rate.
    """
    phat = submissions / views
    return (
        phat
        + z * z / (2 * views)
        - z * np.sqrt((phat * (1 - phat) + z * z / (4 * views)) / views)
    ) / (1 + z * z / views)
