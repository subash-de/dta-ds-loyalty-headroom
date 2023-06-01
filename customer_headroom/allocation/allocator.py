from typing import Optional, Dict, Tuple, List
from pyspark.sql import functions as F, DataFrame, types as T
from functools import partial
from itertools import chain

from dtaml.databricks import get_spark

spark = get_spark()


class Allocator(object):
    def __init__(
            self,
            feature_col: str,
            offer_limits: Dict[str, Tuple[float]],
            user_key: str = "cust_id",
            outlier_min: float = -0.5,
            outlier_max: float = 200.,
            max_increase: float = 80.,
            min_increase: float = 5.,
            headroom_factor: float = 1.2,
            N_offer_fill: int = 2,
            large_offers: Optional[List[int]] = None,
            small_offers: Optional[List[int]] = None,
            fill_offer: Optional[int] = None,
            offer_desc: Optional[Dict[str, str]] = None,
            date_format: Optional[str] = "%Y%m%d",
            prev_not_bought_factor: float = 1,
    ):
        self.feature_col = feature_col
        self.offer_limits = offer_limits
        self.user_key = user_key
        self.outlier_min = outlier_min
        self.outlier_max = outlier_max
        self.max_increase = max_increase
        self.min_increase = min_increase
        self.headroom_factor = headroom_factor
        self.N_offer_fill = N_offer_fill
        if large_offers is None:
            offer_limit_maxes = dict(sorted(self.offer_limits.items(), key=lambda x: max(x[1]),
                                            reverse=True)[:self.N_offer_fill])
            self.large_offers = list((offer_limit_maxes.keys()))
        else:
            self.large_offers = large_offers
        if small_offers is None:
            offer_limit_mins = dict(sorted(self.offer_limits.items(), key=lambda x: min(x[1]),
                                           reverse=False)[:self.N_offer_fill])
            self.small_offers = list((offer_limit_mins.keys()))
        else:
            self.small_offers = small_offers
        if fill_offer:
            self.fill_offer = fill_offer
        else:
            self.fill_offer = min(offer_limits, key=offer_limits.get)
        if offer_desc:
            self.offer_desc = dict(offer_desc)
        else:
            self.offer_desc = None
        self.date_format = date_format
        
        self.prev_not_bought_factor = prev_not_bought_factor
        self.large_lim = max(list(chain(*self.offer_limits.values())))
        self.get_large_offer = F.udf(partial(self.get_offer, offers=self.large_offers), T.IntegerType())
        self.get_small_offer = F.udf(partial(self.get_offer, offers=self.small_offers), T.IntegerType())
        self.get_offer_desc_part = F.udf(partial(self.get_offer_desc, offer_desc=self.offer_desc), T.StringType())

    @staticmethod
    def get_offer(rand, offers):
        try:
            idx = int(rand * len(offers))
            return int(offers[idx])
        except IndexError or KeyError:
            return int(offers[0])

    @staticmethod
    def get_offer_desc(offer, offer_desc):
        if offer_desc:
            return offer_desc[str(offer)]
        else:
            return "Missing"

    def get(self,
            predictions: DataFrame,
            audience: Optional[DataFrame] = None
            ) -> DataFrame:
        """
        Allocate customers from Headroom predictions.
        """
        if audience:
            predictions = (predictions
                           .join(audience
                                 .select(self.user_key)
                                 .distinct(), on=self.user_key, how='right')
                           )

        prediction_scores = self.get_prediction_scores(predictions)

        prediction_scores_tagged = self.tag_outliers(prediction_scores)

        headroom_predictions = self.get_headroom(prediction_scores_tagged)

        headroom_export = self.prepare_export(headroom_predictions)

        return headroom_export

    def get_prediction_scores(self, predictions):
        pred_scores = (predictions
                      #  .withColumn("pct_error",
                      #              100. * (F.col("prediction_out") - F.col(self.feature_col)) / (
                      #                  F.col(self.feature_col)))

                      # set pct error to 0 if there is no purchase (value of 0 in feature col)
                      .withColumn("pct_error", F.when(F.col(self.feature_col) >0, 
                                   100. * (F.col("prediction_out") - F.col(self.feature_col)) / (
                                       F.col(self.feature_col))).otherwise(0)
                                   )
                      #  .groupby(self.user_key, "experian_hh_composition", "segmentation")
                      #  .agg(F.mean(self.feature_col).alias("mean_input"),
                      #       F.sum(self.feature_col).alias("sum_input"),
                      #       F.sum("prediction_out").alias("sum_prediction"),
                      #       F.mean("pct_error").alias("mean_pct_error"),
                      #       (F.sum(F.col(self.feature_col) * F.col("visits")) / F.sum(
                      #           F.col("visits"))).alias("weightedmean_input"),
                      #       (F.sum(F.col("pct_error") * F.col("visits")) / F.sum(F.col("visits"))).alias(
                      #           "weightedmean_pct_error")
                      #       )
                    
                       .withColumn("offer_id", F.lit(None))
                       )
        return pred_scores

    def tag_outliers(self, data):
        data_tagged = (data
                       .withColumn("outlier", F.when(((F.col("pct_error") >= self.outlier_min) &
                                                      (F.col("pct_error") <= self.outlier_max)
                                                      ), 0).otherwise(1))
                       )
        return data_tagged

    def get_headroom(self, data):

        data_hrm = (data
                    .withColumn("used_headroom_frac",
                                F.when((F.col("pct_error") >= self.max_increase) & (F.col("outlier") == 0),
                                       (1. + self.max_increase / 100.))
                                .when((F.col("pct_error") <= self.min_increase) & (F.col("outlier") == 0),
                                      (1. + self.min_increase / 100.))
                                .when((F.col("pct_error") < self.max_increase) &
                                      (F.col("pct_error") > self.min_increase) & (F.col("outlier") == 0),
                                      1. + F.col("pct_error") / 100.)
                                .otherwise(self.headroom_factor)
                                )
                    # .withColumn("total_used_headroom_per_id",
                    #             F.col(self.feature_col) * F.col("used_headroom_frac"))
                    # set the headroom of non purchase to the prediction
                    .withColumn("total_used_headroom_per_id", F.when(F.col(self.feature_col) >0, 
                                                                       F.col(self.feature_col) * F.col("used_headroom_frac")).otherwise(F.col("prediction_out") * self.prev_not_bought_factor)
                                )
                    # .groupby(self.user_key, "experian_hh_composition", "segmentation") # there are null segmentations, which result in random offer being assigned
                    .groupby(self.user_key)
                    .agg(F.sum("total_used_headroom_per_id").alias("total_used_headroom_whole_time_period"),
                    F.sum(self.feature_col).alias("sum_total_spend_whole_period"))
                    .withColumn("sum_total_spend", F.col("sum_total_spend_whole_period") )
                    .withColumn("total_used_headroom", F.col("total_used_headroom_whole_time_period") )
                    .withColumn("rand", F.rand())
                    .withColumn("offer_id", F.lit(None))
                    # .withColumn("total_used_headroom",
                    #             F.col("weightedmean_input") * F.col("used_headroom_frac"))
                    # .withColumn("rand", F.rand())
                    )
        for k, v in self.offer_limits.items():
            offer_id = int(k)
            data_hrm = (data_hrm
                        .withColumn("offer_id", F.when((F.col("total_used_headroom") >= v[0]) &
                                                       (F.col("total_used_headroom") < v[1]), offer_id)
                                    .otherwise(F.col("offer_id"))
                                    )
                        )

        data_out = (data_hrm
                    # If very large headroom. Probably some outliers. For now random spread these offers over the top offer range.
                    .withColumn("offer_id", F.when((F.col("total_used_headroom") >= self.large_lim),
                                                   self.get_large_offer(F.col("rand")))
                                .otherwise(F.col("offer_id")))
                    # If offer Id is still null then an outlier. Give a random small offer.
                    .withColumn("offer_id",
                                F.when((F.col("offer_id").isNull()), self.get_small_offer(F.col("rand")))
                                .otherwise(F.col("offer_id")))
                    .withColumn("desc", self.get_offer_desc_part(F.col("offer_id")))
                    )

        return data_out

    def prepare_export(self, data):
        data_export = (data
                       .withColumn("offer_id", F.when(F.col("offer_id").isNull(), F.lit(self.fill_offer))
                                   .otherwise(F.col("offer_id"))
                                   )
                       .withColumn("spend_plus_headroom", F.round("total_used_headroom", 2))
                       .withColumn("estimated_spend", F.round(F.col("sum_total_spend"), 2))
                       .withColumn("estimated_headroom",
                                   F.round(F.col("spend_plus_headroom") - F.col("sum_total_spend"),
                                           2))
                       .select(self.user_key, "offer_id", "estimated_spend", "estimated_headroom",
                               "spend_plus_headroom", "desc")
                       .dropDuplicates(subset=[self.user_key])
                       )
        return data_export
