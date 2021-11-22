from typing import Any, Optional
import pandas as pd
from surprise import SVD
from surprise import Dataset
import surprise.dataset as surprise_ds
from surprise.model_selection import cross_validate
from surprise.reader import Reader

from scipy import stats
from sklearn import metrics


def build_recommender(
        X: surprise_ds.DatasetAutoFolds,
        method: str,
        build_trainset: bool = True
) -> Any:
    """
    Build Surprise Recommender

    Only SVD is added currently.
    """

    if method.lower() == "svd":
        algorithm = SVD()
    else:
        algorithm = None
    if algorithm:
        if build_trainset:
            X = X.build_full_trainset()
        algorithm.fit(
            X
        )
        return algorithm
    else:
        raise NotImplementedError()
