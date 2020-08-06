from typing import List, Tuple

import pandas as pd


class Processor:
    """The data preprocessing pipeline."""

    def __init__(self) -> None:
        self.features: List[str] = []

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
        if not infer:
            data["completion_rate"] = data["submissions"] / data["views"]
            self.select_features(data)
            return data[self.features], data["completion_rate"]
        return data[self.features]
