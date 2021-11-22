from typing import Any, Iterable, Optional
from pyspark.sql import functions as F, DataFrame, types as T, Window as W
from functools import partial
from dtaml.databricks import get_spark

spark = get_spark()


class Predictor(object):
    def __init__(
            self,
            feature_col: str,
            pred_key: str = "l2_id",
            pred_items: Iterable = ("01", "02", "03", "04", "05", "07"),
            user_key: str = "cust_id",
            date_format: Optional[str] = "%Y%m%d",
            lognorm: bool = True,
            min_col: Optional[float] = None,
            max_col: Optional[float] = None
    ):
        self.feature_col = feature_col
        self.pred_key = pred_key
        self.pred_items = pred_items
        self.user_key = user_key
        self.date_format = date_format
        self.lognorm = lognorm
        self.min_col = 0 if not min_col else min_col
        self.max_col = 0 if not max_col else max_col

    def get(self,
            data: DataFrame,
            algo: Any,
            predict_col: str = "prediction"
            ) -> DataFrame:
        """
        Predict customers' Headroom using provided (Surprise) algo.
        """
        out_predict_col = f"{predict_col}_out"
        pred_data = self._build_pred_data(data)
        predictions = self.predict(pred_data, algo, self.user_key, self.pred_key)
        predictions_transformed = self.tranform_predict(predictions,
                                                        predict_col=predict_col,
                                                        out_predict_col=out_predict_col)
        predictions_headroom = self.get_headroom(data=data,
                                                 pred_data=predictions_transformed,
                                                 predict_col=out_predict_col
                                                 )
        return predictions_headroom

    def _build_pred_data(self, df: DataFrame) -> DataFrame:
        """
        Build prediction dataset, every L2 for every customer (regardless of previous purchases).
        """
        predictor_items = spark.createDataFrame(map(lambda x: [x], self.pred_items), [self.pred_key])
        pred_df = (df.select(self.user_key).distinct()
                   .join(F.broadcast(predictor_items))
                   )
        return pred_df

    @staticmethod
    def predict(
            df: DataFrame,
            algorithm: Any,
            user_key: str,
            pred_key: str,
            predict_col: str = "prediction"
    ) -> DataFrame:
        """
        Predict from Recommendation Engine

        Needs to be a static method since:
        'Exception: SparkContext should only be created and accessed on the driver.'
        """

        def _predict(k, v, algorithm):
            y = algorithm.predict(uid=str(k), iid=str(v)).est
            return float(y)

        predict_recommendation_udf = F.udf(partial(_predict, algorithm=algorithm), T.DoubleType())
        pred_df = df.withColumn(predict_col, predict_recommendation_udf(user_key, pred_key))

        return pred_df

    def tranform_predict(self,
                         pred_df: DataFrame,
                         predict_col: str = "prediction",
                         out_predict_col: str = "prediction_out"
                         ) -> DataFrame:
        """
        Transform the lognorm column back to original
        """
        if self.lognorm:
            data_transform = pred_df.withColumn(out_predict_col,
                                                (F.exp(F.col(predict_col) + self.min_col) * (
                                                            self.max_col - self.min_col)))
        else:
            data_transform = pred_df.withColumn(out_predict_col, F.col(predict_col))
        return data_transform

    def get_headroom(self,
                     data: DataFrame,
                     pred_data: DataFrame,
                     predict_col: str = "prediction_out"
                     ):
        """
        Find Headroom, Actual - Predicted [Metric e.g. number_of_transactions] in category.
        Also provide the headroom rank, 1 being the most opportunity for headroom in category.
        """
        data_headroom = (data.join(pred_data, on=[self.user_key, self.pred_key], how="left")
                         .fillna(0, subset=[self.feature_col])
                         .withColumn("headroom", F.col(self.feature_col) - F.col(predict_col))
                         .withColumn("headroom_item_rank", F.dense_rank().over(W.partitionBy(self.pred_key)
                                                                               .orderBy(F.col("headroom"))))
                         )
        return data_headroom
