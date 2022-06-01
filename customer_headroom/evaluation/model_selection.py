from typing import Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import math
from functools import partial
from sklearn.model_selection import train_test_split
from dtaml.logging import get_logger
from dtaml.databricks import get_spark

import seaborn as sns
from surprise.model_selection import KFold
from scipy import stats
from sklearn import metrics

sns.set(style="whitegrid")
logger = get_logger("customer-headroom")
spark = get_spark()


class Evaluator(object):
    def __init__(
            self,
            algorithm: Any,
            data_processor: Any,
            user_key: str,
            pred_key: str,
            pred_items: str,
            dev_size: float,
            test_size: float,
            split_col: str,
            sample: Optional[int] = None,
            random_state: Optional[int] = None
    ):
        self.algorithm = algorithm
        self.data_processor = data_processor
        self.user_key = user_key
        self.pred_key = pred_key
        self.pred_items = pred_items
        self.dev_size = dev_size
        self.test_size = test_size
        self.split_col = split_col
        self.sample = sample
        self.random_state = random_state

    def _train_dev_test_split(
            self,
            data: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Splitting `data` into train, dev and test sets
        split is randomised by `self.split_col` e.g. cust_id
        """
        # since there could be more than 1 purchase mission for a customer,
        # we need to split the sets by `self.split_col`
        unique_ids = data.loc[:, self.split_col].unique()
        train_ids, test_ids = train_test_split(
            unique_ids, test_size=self.test_size, random_state=self.random_state
        )
        train_ids, dev_ids = train_test_split(
            train_ids, test_size=self.dev_size / (1 - self.test_size), random_state=self.random_state
        )

        train_data = data.loc[data.loc[:, self.split_col].isin(train_ids), :]
        dev_data = data.loc[data.loc[:, self.split_col].isin(dev_ids), :]
        test_data = data.loc[data.loc[:, self.split_col].isin(test_ids), :]

        assert len(set(train_data.index) & set(dev_data.index)) == 0, \
            "There is overlap between train and dev sets"
        assert len(set(train_data.index) & set(test_data.index)) == 0, \
            "There is overlap between train and test sets"
        assert len(set(test_data.index) & set(dev_data.index)) == 0, \
            "There is overlap between dev and test sets"
        logger.info(
            f"training set size: {train_data.shape}, "
            f"dev set size: {dev_data.shape}, "
            f"test set size: {test_data.shape}"
        )
        return train_data, dev_data, test_data

    def _process_data(
            self,
            data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform data using the loaded in data_processor object
        """
        rec_data = self.data_processor.transform(data)
        return rec_data

    def _preprocess_dataframes(
            self,
            train_data: pd.DataFrame,
            dev_data: pd.DataFrame,
            test_data: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Preprocess train/dev/test sets
        """
        logger.info("Preprocessing training data")
        train_data_prep = self._process_data(data=train_data).build_full_trainset()
        logger.info("Preprocessing dev data")
        dev_data_prep = self._process_data(data=dev_data).build_full_trainset()
        logger.info("Preprocessing test data")
        test_data_prep = self._process_data(data=test_data).build_full_trainset()

        return train_data_prep, dev_data_prep, test_data_prep

    def _build_predict_dataframe(self,
                                 data_idx: List[Tuple],
                                 full_range: bool = False
                                 ) -> pd.DataFrame:
        """
        Build Prediciton Datasets from the List of Tuples output from the surprise Datasets.
        """
        data_tuples = data_idx
        columns = [self.user_key, self.pred_key, "val"]

        pred_dataframe = pd.DataFrame(data_tuples, columns=columns)
        if full_range:
            accs = {acc[0] for acc in data_tuples}
            pred_dataframe = pred_dataframe.merge(
                pd.DataFrame([(acc, item) for acc in accs for item in self.pred_items],
                             columns=[self.user_key, self.pred_key]),
                on=[self.user_key, self.pred_key], how="outer")
        return pred_dataframe

    def _predict(self,
                 algorithm: Any,
                 data_tuple: Tuple,
                 full_range: bool = False
                 ) -> pd.DataFrame:
        """
        Find algorithm predictions from the data tuples -> pandas Dataframes.
        """

        data_ = self._build_predict_dataframe(data_tuple, full_range=full_range)

        data_["prediction"] = data_.apply(
            lambda row: algorithm.predict(uid=str(row[self.user_key]), iid=str(row[self.pred_key])).est, axis=1)
        return data_

    def _predict_set(
            self,
            algorithm: Any,
            train_data: Tuple,
            dev_data: Tuple,
            test_data: Tuple,
            full_range: bool = False
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Make predictions on train/dev/test sets
        """

        train_data_ = self._predict(algorithm, train_data.build_testset(), full_range=full_range)
        dev_data_ = self._predict(algorithm, dev_data.build_testset(), full_range=full_range)
        test_data_ = self._predict(algorithm, test_data.build_testset(), full_range=full_range)

        return train_data_, dev_data_, test_data_

    def evaluate(self,
                 data: pd.DataFrame,
                 run_tag: Optional[str] = None
                 ) -> pd.DataFrame:
        """
        Evaluate model performance with train/dev/test split
        """
        if self.sample:
            logger.info(f"Apply Sample by Factor of {self.sample}")
            data = data.sample(frac=self.sample, axis=0., random_state=self.random_state)

        logger.info("Splitting `data` into train/dev/test")
        train_data, dev_data, test_data = self._train_dev_test_split(data)

        logger.info("Preprocessing train/dev/test datasets")
        train_data_prep, dev_data_prep, test_data_prep = \
            self._preprocess_dataframes(train_data, dev_data, test_data)

        logger.info("Building Recommender Algorithm with Training Data")
        algo = self.algorithm(train_data_prep, build_trainset=False)

        logger.info("Get train/dev/test Predictions")
        train_pred, dev_pred, test_pred = self._predict_set(algo,
                                                            train_data=train_data_prep,
                                                            dev_data=dev_data_prep,
                                                            test_data=test_data_prep,
                                                            full_range=True)

        logger.info("Evaluate train/dev/test Datasets")
        evaluations = self._evaluate_set(algo,
                                         train_pred, dev_pred, test_pred)
        if run_tag:
            evaluations.loc[:, "run"] = run_tag

        return evaluations

    def evaluate_kfold(self,
                       data: pd.DataFrame,
                       n_splits: int = 3,
                       shuffle: bool = False,
                       run_tag: Optional[str] = None
                       ) -> pd.DataFrame:
        """
        Evaluate model performance with K-Fold
        """
        if self.sample:
            logger.info(f"Apply Sample by Factor of {self.sample}")
            data = data.sample(frac=self.sample, axis=0., random_state=self.random_state)

        logger.info("Construct K-FOLDs")
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)
        logger.info("Preprocessing datasets")
        data_ = self._process_data(data)
        evaluations = pd.DataFrame()
        for i, (train_idx, test_idx) in enumerate(kf.split(data_)):
            logger.info(f"Evaluate KFOLD Dataset - k={i}")
            algo = self.algorithm(train_idx, build_trainset=False)
            train_pred = self._predict(algo, train_idx.build_testset(), full_range=True)
            test_pred = self._predict(algo, test_idx, full_range=True)
            train_res = self._evaluate(algo, train_pred, tag=f"train-k{i}")
            test_res = self._evaluate(algo, test_pred, tag=f"test-k{i}")
            evaluations = pd.concat([evaluations, train_res, test_res])
        if run_tag:
            evaluations.loc[:, "run"] = run_tag
        return evaluations

    def _evaluate(self,
                  algorithm: Any,
                  pred_data: pd.DataFrame,
                  tag: Optional[str] = None
                  ) -> pd.DataFrame:
        """
        Run evaluations on the prediction pandas DataFrame.
        Current metrics are:
         - Root-Mean-Square Error (RMSE)
         - Mean-Square Error (MAE)
         -  AVERAGE RECIPROCAL HIT RANK (ARHR)
        """

        results = {}
        if tag:
            results["tag"] = tag

        # RSME and MAE
        actual = pred_data[~np.isnan(pred_data["val"])]["val"]
        pred = pred_data[~np.isnan(pred_data["val"])]["prediction"]
        results["rsme"] = math.sqrt(metrics.mean_squared_error(actual, pred))
        results["mae"] = metrics.mean_absolute_error(actual, pred)

        # AVERAGE RECIPROCAL HIT RANK (ARHR)
        results["arhr"] = self.get_arhr(algorithm, pred_data)

        results_df = pd.DataFrame({k: [v] for (k, v) in results.items()})
        return results_df

    def _evaluate_set(
            self,
            algorithm: Any,
            train_pred: pd.DataFrame,
            dev_pred: pd.DataFrame,
            test_pred: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run Evaluation function on the train/dev/test datasets.
        """

        train_res = self._evaluate(algorithm, train_pred, tag="train")
        dev_res = self._evaluate(algorithm, dev_pred, tag="dev")
        test_res = self._evaluate(algorithm, test_pred, tag="test")

        results = pd.concat([train_res, dev_res, test_res])

        return results

    def get_arhr(self,
                 algorithm: Any,
                 df: pd.DataFrame) -> float:
        """
        Calculate the AVERAGE RECIPROCAL HIT RANK (ARHR) Metric from the input DataFrame given recommendation Algorithm
        """
        get_true_vals_part = partial(self.get_true_vals, catCols=self.pred_items)
        get_sim_scores_part = partial(self.get_sim_scores, algorithm=algorithm, catCols=self.pred_items)
        cust_vals = df.groupby(self.user_key).apply(lambda x: dict(zip(x[self.pred_key], x['val']))).reset_index(
            name='mapping')
        cust_vals["vals"] = cust_vals.apply(lambda x: get_true_vals_part(x['mapping']), axis=1)
        cust_vals["scores"] = cust_vals.apply(lambda x: get_sim_scores_part(x['cust_id']), axis=1)
        cust_vals["rank_vals"] = cust_vals.apply(lambda x: self.get_ranks(x["vals"]), axis=1)
        cust_vals["rank_scores"] = cust_vals.apply(lambda x: self.get_ranks(x["scores"]), axis=1)
        arhr = self.arhr_metric(cust_vals)
        return arhr

    @staticmethod
    def get_true_vals(row, catCols):
        """
        Extract True Values from the given pandas DataFrame Row.
        """
        vals = []
        for v in catCols:
            vals.append(row.get(v))
        return vals

    @staticmethod
    def get_sim_scores(cust, algorithm, catCols):
        """
        Get Recommendation scores as list
        """
        preds = []
        for v in catCols:
            preds.append(algorithm.predict(uid=cust, iid=v).est)
        return preds

    @staticmethod
    def get_ranks(row):
        """
        Get Ranks from input lists of scores/values.
        """
        values = np.array([x if x is not None else 0 for x in row])
        ranks = stats.rankdata(-values, method="max")
        return [r if v != 0 else np.nan for r, v in zip(ranks, values)]

    @staticmethod
    def arhr_metric(df: pd.DataFrame) -> float:
        """
        AVERAGE RECIPROCAL HIT RANK (ARHR)
        source: https://medium.com/fnplus/evaluating-recommender-systems-with-python-code-ae0c370c90be
        """

        def get_reciprocal_rank_sum(row):
            row_test = row["rank_vals"]
            row_predict = row["rank_scores"]
            row_bool = [False if np.isnan(i) else True for i in row_test]
            return (1. / np.asarray(row_predict)[row_bool]).sum()

        df["reciprocal_sum_rank"] = df.apply(get_reciprocal_rank_sum, axis=1)
        return 1 / (len(df)) * df["reciprocal_sum_rank"].sum()
