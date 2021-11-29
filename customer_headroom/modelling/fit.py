from typing import Any
import surprise
# from surprise import Dataset
# from surprise.model_selection import cross_validate
# from surprise.reader import Reader
#
# from scipy import stats
# from sklearn import metrics


def build_recommender(
        X: Any,
        method: str,
        build_trainset: bool = True
) -> Any:
    """
    Build Surprise Recommender

    Only SVD is added currently.
    """

    if method.lower() == "svd":
        algorithm = surprise.SVD()
    elif method.lower() == "svdpp":
        algorithm = surprise.SVDpp()
    elif method.lower() == "nmf":
        algorithm = surprise.NMF()
    elif method.lower() == "knn":
        algorithm = surprise.KNNBasic()
    elif method.lower() == "knn_zscore":
        algorithm = surprise.KNNWithZScore()
    elif method.lower() == "knn_mean":
        algorithm = surprise.KNNWithMeans()
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
