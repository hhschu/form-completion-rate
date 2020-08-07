"""Request and response models."""
from typing import List

from pydantic import BaseModel


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


class Version(BaseModel):
    """Version number of the model."""

    version: int


class Versions(BaseModel):
    """Version numbers of available models."""

    versions: List[int]


class Score(BaseModel):
    """Probablity of submition."""

    score: float
