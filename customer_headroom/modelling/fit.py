from typing import Any, Dict, Optional

import surprise
from surprise.model_selection import GridSearchCV, KFold


def update_parameters(algo: any, param_dict: Dict[str, Any]) -> Any:
    for k, v in param_dict.items():
        algo.__setattr__(k, v)
    return algo


def build_recommender(
    X: Any,
    method: str,
    build_trainset: bool = True,
    measure: str = "rmse",
    n_splits: int = 3,
    shuffle: bool = True,
    random_state: int = 42,
    params: Optional[Dict[str, Any]] = None,
    param_grid: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Build Surprise Recommender

    Only SVD is added currently.
    """

    if method.lower() == "svd":
        algo_type = surprise.SVD
    elif method.lower() == "svdpp":
        algo_type = surprise.SVDpp
    elif method.lower() == "nmf":
        algo_type = surprise.NMF
    elif method.lower() == "knn":
        algo_type = surprise.KNNBasic
    elif method.lower() == "knn_zscore":
        algo_type = surprise.KNNWithZScore
    elif method.lower() == "knn_mean":
        algo_type = surprise.KNNWithMeans
    else:
        algo_type = None

    if algo_type:
        algorithm = algo_type()

        if param_grid and ((params is None) or (params == "None")):
            # Find optimal parameters for this group.
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            gs = GridSearchCV(algo_type, param_grid, measures=[measure], cv=kf)
            gs.fit(X)
            params = gs.best_params[measure]
            algorithm = update_parameters(algorithm, params)
        elif params and not ((params is None) or (params == "None")):
            # Use provided parameters.
            algorithm = update_parameters(algorithm, params)

        if build_trainset:
            # Build the training dataset
            X = X.build_full_trainset()

        algorithm.fit(X)
        params_used = algorithm.__dict__
        return algorithm, params_used
    else:
        raise NotImplementedError()
