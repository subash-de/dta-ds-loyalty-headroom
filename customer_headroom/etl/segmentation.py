from datetime import datetime, timedelta
from functools import reduce
from typing import Optional, List, Dict, Tuple
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from pyspark.sql import functions as F, types as T, DataFrame
from pyspark.ml.feature import PCA as sparkPCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler

from customer_headroom.etl.build_dataset import BaseManager
from dtaml.logging import get_logger

logger = get_logger("customer-headroom")
sns.set_style("darkgrid")


class SegmentationDataManager(BaseManager):
    def __init__(
            self,
            etl_date: int,
            lookback_days: int = 365,
            l1_id: str = "FD",
            user_id: str = "cust_id",
            date_format: Optional[str] = "%Y%m%d"
    ):
        self.etl_date = int(etl_date)
        self.lookback_days = lookback_days
        self.lookback_date = int((datetime.strptime(str(etl_date), date_format) -
                                  timedelta(days=lookback_days)).strftime(date_format))
        self.l1_id = l1_id
        self.user_id = user_id

        self.date_format = date_format

    def get(self,
            trx_line: DataFrame,
            sparks_custs: DataFrame,
            cust_master: DataFrame,
            customer_input: Optional[DataFrame] = None
            ) -> DataFrame:
        """
        Entry Method for SegmentationDataManager. This triggers the full build of the segmentation Dataset.

        trx_line, sparks_custs and cust_master are factory tables.
        customer_input is an optional customer input table used only for selecting particular customers
         to be included in the output DataFrame.
        """

        spent_in_period = self._find_spend_in_period(trx_line, sparks_custs, customer_input)

        sparks_custs_spent_in_period = self._sparks_customer_mosaic_data(spent_in_period, cust_master)

        clean_demog = self.clean_demog_data(sparks_custs_spent_in_period)

        data_out = self._prepare_dataset(clean_demog)

        return data_out

    def _find_spend_in_period(self,
                              trx_line: DataFrame,
                              sparks_custs: DataFrame,
                              customer_input: Optional[DataFrame] = None
                              ) -> DataFrame:
        """
        This method performs date filtering on the all_transaction_line factory table to extract transactions
        that occur in the date range specified by the init etl_date and lookback_days.
        The sparks_custs DataFrame is required to may to the desired customer/user key.
        """
        # TODO: Hardcoded at l1 level. Could expand in the future to other levels within FD/GM.
        l1_ids = ["FD", "GM"]
        trx_line_with_date = self._add_date(trx_line)

        trx_line_between_date = (trx_line_with_date
                                 .filter(self.get_common_filters())
                                 .filter(F.col("date").between(self.lookback_date, self.etl_date))
                                 .join(sparks_custs.select("account_id", "cust_id").distinct(), on="cust_id")
                                 )

        if customer_input is not None:
            trx_line_between_date = (trx_line_between_date
                                     .join(customer_input.select(self.user_id).distinct(),
                                           on=self.user_id, how="inner")
                                     )

        spent_in_period = (trx_line_between_date
                           .groupby(self.user_id)
                           .pivot("l1_id", l1_ids)
                           .agg(F.sum("sales_amt").alias("total_spend"),
                                F.count("BASKET_ID").alias("total_items"),
                                F.countDistinct("BASKET_ID").alias("total_baskets"))
                           .fillna(0, subset=[f"{l1}_{val}" for val in ["total_spend",
                                                                        "total_items",
                                                                        "total_baskets"] for l1 in l1_ids])
                           .withColumn("total_spend", F.col("FD_total_spend") + F.col("GM_total_spend"))
                           .withColumn("total_items", F.col("FD_total_items") + F.col("GM_total_items"))
                           .withColumn("total_baskets", F.col("FD_total_baskets") + F.col("GM_total_baskets"))
                           .withColumn("FD_spend", F.col("FD_total_spend") / F.col("total_spend"))
                           .withColumn("GM_spend", F.col("GM_total_spend") / F.col("total_spend"))
                           .withColumn("FD_items", F.col("FD_total_items") / F.col("total_items"))
                           .withColumn("GM_items", F.col("GM_total_items") / F.col("total_items"))
                           .withColumn("FD_baskets", F.col("FD_total_baskets") / F.col("total_baskets"))
                           .withColumn("GM_baskets", F.col("GM_total_baskets") / F.col("total_baskets"))
                           .filter(F.col(f"{self.l1_id}_total_spend") >= 0.1)
                           )

        return spent_in_period

    def _sparks_customer_mosaic_data(self,
                                     spent_in_period: DataFrame,
                                     cust_master: DataFrame
                                     ) -> DataFrame:
        """
        Join on to sparks customer transaction DataFrame the experian mosaic features from the cust_master
        factory table.
        """
        sparks_custs_spent_in_period = (spent_in_period
                                        # Changed to Left join. All customer master features are Null,
                                        # including the experian mosaics.
                                        .join(cust_master, on="cust_id", how="left")
                                        )
        return sparks_custs_spent_in_period

    def clean_demog_data(self,
                         sparks_custs_spent_in_period: DataFrame):
        """

        """
        cleaned_demog_data = (sparks_custs_spent_in_period
                              # take lowest age as the estimate - may want to improve
                              .withColumn("experian_age_n", F.substring("experian_age", 1, 2))
                              .withColumn("age", F.when(F.col("age").between(18, 120), F.col("age")).otherwise(None))
                              .withColumn("experian_affluence",
                                          F.coalesce(F.col("experian_affluence").cast("integer"), F.lit(5)))
                              # 56 as the average - may need to be restimated
                              .withColumn("age_to_use", F.coalesce(F.col("age"), F.col("experian_age_n"), F.lit(56)))
                              .withColumn("banded_age", F
                                          .when(F.col("age_to_use").between(18, 25), "18-25")
                                          .when(F.col("age_to_use").between(26, 30), "26-30")
                                          .when(F.col("age_to_use").between(31, 35), "31-35")
                                          .when(F.col("age_to_use").between(36, 40), "36-40")
                                          .when(F.col("age_to_use").between(41, 45), "41-45")
                                          .when(F.col("age_to_use").between(46, 50), "46-50")
                                          .when(F.col("age_to_use").between(51, 55), "51-55")
                                          .when(F.col("age_to_use").between(56, 60), "56-60")
                                          .when(F.col("age_to_use").between(61, 65), "61-65")
                                          .when(F.col("age_to_use").between(66, 70), "66-70")
                                          .when(F.col("age_to_use").between(71, 75), "71-75")
                                          .when(F.col("age_to_use").between(76, 120), "76+")
                                          .otherwise(None))
                              )
        return cleaned_demog_data

    def _prepare_dataset(self,
                         df: DataFrame,
                         prefix: str = "Cat_"
                         ) -> DataFrame:
        """
        experian_hh_composition: Household Composition is a household level demographic variable that identifies the type of family living at an address.
        experian_mosaic_uk_type: Mosaic UK is a geodemographic classification that paints a rich picture of UK consumers in terms of socio-demographics,
                                 lifestyles and behaviour, and provides a detailed understanding of UK society. It is an essential tool for any organisation
                                 that wishes to understand more about UK consumers, and develop successful marketing solutions that are tailored to the needs
                                 of the UK marketplace. Mosaic UK classifies all UK consumers into 66 distinct lifestyle types and 15 groups which
                                 comprehensively describe their socio-economic and socio-cultural behaviour.
        """
        df = df.select("cust_id",
                       F.col("age_to_use").cast("int").alias("age_to_use"),
                       F.coalesce(F.col("gender"), F.lit("U")).alias("gender"),
                       F.concat(F.lit(prefix), F.coalesce(F.col("experian_hh_composition"),
                                                          F.lit("U"))).alias("experian_hh_composition"),
                       F.concat(F.lit(prefix), F.coalesce(F.col("experian_mosaic_uk_type"),
                                                          F.lit("U"))).alias("experian_mosaic_uk_type"),
                       F.when(F.col("gender") == "F", 1).otherwise(0).alias("gender_F"),
                       F.when(F.col("gender") == "M", 1).otherwise(0).alias("gender_M"),
                       F.when(F.col("experian_hh_composition") == f"{prefix}00", 1).otherwise(0).alias("hh_comp_00"),
                       F.when(F.col("experian_hh_composition") == f"{prefix}01", 1).otherwise(0).alias("hh_comp_01"),
                       F.when(F.col("experian_hh_composition") == f"{prefix}02", 1).otherwise(0).alias("hh_comp_02"),
                       F.when(F.col("experian_hh_composition") == f"{prefix}03", 1).otherwise(0).alias("hh_comp_03"),
                       F.when(F.col("experian_hh_composition") == f"{prefix}04", 1).otherwise(0).alias("hh_comp_04"),
                       F.when(F.col("experian_hh_composition") == f"{prefix}05", 1).otherwise(0).alias("hh_comp_05"),
                       F.when(F.col("experian_hh_composition") == f"{prefix}06", 1).otherwise(0).alias("hh_comp_06"),
                       F.when(F.col("experian_hh_composition") == f"{prefix}07", 1).otherwise(0).alias("hh_comp_07"),
                       F.when(F.col("experian_hh_composition") == f"{prefix}08", 1).otherwise(0).alias("hh_comp_08"),
                       F.when(F.col("experian_hh_composition") == f"{prefix}09", 1).otherwise(0).alias("hh_comp_09"),
                       F.when(F.col("experian_hh_composition") == f"{prefix}10", 1).otherwise(0).alias("hh_comp_10"),
                       F.when(F.col("experian_hh_composition") == f"{prefix}12", 1).otherwise(0).alias("hh_comp_11"),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}00', 1).otherwise(0).alias('mosaic_00'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}01', 1).otherwise(0).alias('mosaic_01'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}02', 1).otherwise(0).alias('mosaic_02'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}03', 1).otherwise(0).alias('mosaic_03'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}04', 1).otherwise(0).alias('mosaic_04'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}05', 1).otherwise(0).alias('mosaic_05'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}06', 1).otherwise(0).alias('mosaic_06'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}07', 1).otherwise(0).alias('mosaic_07'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}08', 1).otherwise(0).alias('mosaic_08'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}09', 1).otherwise(0).alias('mosaic_09'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}10', 1).otherwise(0).alias('mosaic_10'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}11', 1).otherwise(0).alias('mosaic_11'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}12', 1).otherwise(0).alias('mosaic_12'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}13', 1).otherwise(0).alias('mosaic_13'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}14', 1).otherwise(0).alias('mosaic_14'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}15', 1).otherwise(0).alias('mosaic_15'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}16', 1).otherwise(0).alias('mosaic_16'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}17', 1).otherwise(0).alias('mosaic_17'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}18', 1).otherwise(0).alias('mosaic_18'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}19', 1).otherwise(0).alias('mosaic_19'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}20', 1).otherwise(0).alias('mosaic_20'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}21', 1).otherwise(0).alias('mosaic_21'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}22', 1).otherwise(0).alias('mosaic_22'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}23', 1).otherwise(0).alias('mosaic_23'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}24', 1).otherwise(0).alias('mosaic_24'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}25', 1).otherwise(0).alias('mosaic_25'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}26', 1).otherwise(0).alias('mosaic_26'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}27', 1).otherwise(0).alias('mosaic_27'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}28', 1).otherwise(0).alias('mosaic_28'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}29', 1).otherwise(0).alias('mosaic_29'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}30', 1).otherwise(0).alias('mosaic_30'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}31', 1).otherwise(0).alias('mosaic_31'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}32', 1).otherwise(0).alias('mosaic_32'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}33', 1).otherwise(0).alias('mosaic_33'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}34', 1).otherwise(0).alias('mosaic_34'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}35', 1).otherwise(0).alias('mosaic_35'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}36', 1).otherwise(0).alias('mosaic_36'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}37', 1).otherwise(0).alias('mosaic_37'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}38', 1).otherwise(0).alias('mosaic_38'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}39', 1).otherwise(0).alias('mosaic_39'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}40', 1).otherwise(0).alias('mosaic_40'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}41', 1).otherwise(0).alias('mosaic_41'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}42', 1).otherwise(0).alias('mosaic_42'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}43', 1).otherwise(0).alias('mosaic_43'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}44', 1).otherwise(0).alias('mosaic_44'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}45', 1).otherwise(0).alias('mosaic_45'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}46', 1).otherwise(0).alias('mosaic_46'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}47', 1).otherwise(0).alias('mosaic_47'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}48', 1).otherwise(0).alias('mosaic_48'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}49', 1).otherwise(0).alias('mosaic_49'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}50', 1).otherwise(0).alias('mosaic_50'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}51', 1).otherwise(0).alias('mosaic_51'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}52', 1).otherwise(0).alias('mosaic_52'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}53', 1).otherwise(0).alias('mosaic_53'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}54', 1).otherwise(0).alias('mosaic_54'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}55', 1).otherwise(0).alias('mosaic_55'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}56', 1).otherwise(0).alias('mosaic_56'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}57', 1).otherwise(0).alias('mosaic_57'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}58', 1).otherwise(0).alias('mosaic_58'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}59', 1).otherwise(0).alias('mosaic_59'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}60', 1).otherwise(0).alias('mosaic_60'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}61', 1).otherwise(0).alias('mosaic_61'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}62', 1).otherwise(0).alias('mosaic_62'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}63', 1).otherwise(0).alias('mosaic_63'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}64', 1).otherwise(0).alias('mosaic_64'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}65', 1).otherwise(0).alias('mosaic_65'),
                       F.when(F.col('experian_mosaic_uk_type') == f'{prefix}66', 1).otherwise(0).alias('mosaic_66'),
                       "FD_spend", "GM_spend", "FD_items", "GM_items", "FD_baskets", "GM_baskets")
        return df


class SegmentationManager(BaseManager):
    def __init__(
            self,
            num_cols: List[str] = ["age_to_use", "gender_F", "gender_M",
                                   "FD_spend", "GM_spend", "FD_items",
                                   "GM_items", "FD_baskets", "GM_baskets"],
            cat_cols: List[str] = ["experian_hh_composition"],
            frac_lim: float = 0.1,
            fail_limit: int = 3,
            k_search_min: int = 2,
            k_search_max: int = 8,
            pca_k: Optional[int] = 3,
            data_lower_lim: int = 100,
            user_id: str = "cust_id",
            verbose: bool = True,
            date_format: Optional[str] = "%Y%m%d"
    ):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.frac_lim = frac_lim
        self.fail_limit = fail_limit
        self.k_search_min = k_search_min
        self.k_search_max = k_search_max
        self.pca_k = pca_k
        self.data_lower_lim = data_lower_lim
        self.user_id = user_id
        self.verbose = verbose
        self.date_format = date_format

    def get(self,
            data: DataFrame,
            etl_data: Optional[DataFrame] = None
            ) -> DataFrame:
        """
        Entry method to run SegmentationManager
        SegmentationManager: Segementation input dataset created by SegmentationDataManager.
        etl_data: Optional customer input dataset.
        """
        self.category_list = self._get_demog_cats(data)

        best_kmeans_dict, _all_silhouette_score = self._find_best_KMeans(data)

        data_clustered = self._join_all_best_KMeans(kmeans_dict=best_kmeans_dict)

        if etl_data is not None:
            data_clustered = data_clustered.join(etl_data, on=self.user_id)

        return data_clustered

    def log(self, string: str):
        if self.verbose:
            logger.info(string)

    def _get_category_filters(self,
                              demog_groups: List[str]
                              ) -> DataFrame:
        """
        method to collect category filters used for segmentations.
        """
        demog_cat_filters_str = [f"(F.col('{dc}') == '{dc_group}')" for (dc, dc_group)
                                 in zip(self.cat_cols, demog_groups)]
        demog_cat_filters = eval(" & ".join(demog_cat_filters_str))
        return demog_cat_filters

    def _find_best_KMeans(self,
                          data: DataFrame
                          ) -> Dict[str, DataFrame]:
        """
        This method loops through all pre-defined category segements and runs the KMeans clustering algorithm.
        feature stadardisation is also performed. The spark version of PCA is also an optio however this does not appear
        to be working correctly at the moment.
        For each category we loop through k = [k_search_min to k_search_max] and keep the best performing
        clusters as determined by the silhouette scores.
        The output is a dictionary of DataFrames with the predefined categories as keys.
        """

        count_lim = 0
        all_silhouette_score = {}
        all_silhouette_score_max = {}
        output_dict = {}
        for demog_groups in self.category_list:
            demog_groups_str = str(demog_groups)
            demog_cat_filters = self._get_category_filters(demog_groups)
            data_ = data.filter(demog_cat_filters)
            data_count = data_.count()
            self.log(f"{demog_groups_str}: Count: {data_count}")
            if data_count > self.data_lower_lim:

                assemble = VectorAssembler(inputCols=self.num_cols, outputCol='features')
                assembled_data = assemble.transform(data_)

                output_col = "standardized"
                scale = StandardScaler(inputCol='features', outputCol="standardized")
                data_scale = scale.fit(assembled_data)
                data_scale_output = data_scale.transform(assembled_data)

                if self.pca_k:
                    self.log(f"{demog_groups_str}: \t Use PCA Features")
                    output_col = "pca_features"
                    pca = sparkPCA(k=self.pca_k, inputCol="standardized", outputCol=output_col)
                    pca_model = pca.fit(data_scale_output)
                    data_output = pca_model.transform(data_scale_output)
                else:
                    data_output = data_scale_output

                silhouette_score = {}
                outputs = {}
                evaluator = ClusteringEvaluator(predictionCol='segmentation', featuresCol=output_col,
                                                metricName='silhouette', distanceMeasure='squaredEuclidean')
                fail_count = 0
                for k in range(self.k_search_min, self.k_search_max):

                    KMeans_algo = KMeans(featuresCol=output_col, k=k, predictionCol='segmentation')
                    KMeans_fit = KMeans_algo.fit(data_output)
                    output = KMeans_fit.transform(data_output)

                    fail_tab = (output
                                .groupby("segmentation")
                                .agg(F.count("*").alias("count"),
                                     (F.count("*") / data_count).alias("fraction"))
                                .withColumn("count_fail", F.when(F.col("count") < count_lim, 1).otherwise(0))
                                .withColumn("fraction_fail", F.when(F.col("fraction") < self.frac_lim, 1).otherwise(0))
                                .withColumn("fail", F.greatest(F.col("count_fail"), F.col("fraction_fail")))
                                )
                    fail_check = fail_tab.select(F.max("fail")).rdd.map(lambda x: x[0]).first()

                    if fail_check == 1:
                        fail_count += 1
                        self.log(f"""{demog_groups_str} - {k}:
                        FAILED CHECK: {fail_count} of {self.fail_limit} fails in a row.""")
                        if fail_count >= self.fail_limit:
                            break
                    else:
                        # Reset fail count
                        fail_count = 0
                        outputs[k] = output

                        score = evaluator.evaluate(output)

                        silhouette_score[k] = score
                        self.log(f"{demog_groups_str} - {k}: \t Silhouette Score: {score}")

                all_silhouette_score[demog_groups_str] = silhouette_score
                max_silhouette_score_key = max(silhouette_score, key=silhouette_score.get)
                max_silhouette_score = silhouette_score[max_silhouette_score_key]
                all_silhouette_score_max[demog_groups_str] = max_silhouette_score
                output_dict[demog_groups_str] = outputs[max_silhouette_score_key]
                self.log(
                    f"{demog_groups_str} - K={max_silhouette_score_key} Max Silhouette Score: {max_silhouette_score}")

        return output_dict, all_silhouette_score_max

    def _join_all_best_KMeans(self,
                              kmeans_dict: Dict[str, DataFrame]
                              ) -> DataFrame:
        """
        The dictionary output from self._find_best_KMeans() to concatenated here.
        """
        final_output = reduce(DataFrame.union, kmeans_dict.values())
        for p in range(self.pca_k):
            extract_element = F.udf(lambda v: float(v[p]))
            final_output = final_output.withColumn(f"pca_{p}", extract_element("pca_features"))

        return final_output

    def _get_demog_cats(self,
                        data: DataFrame
                        ) -> DataFrame:
        """
        This method gets the [self.cat_cols] demographic data from the input dataset.
        """
        category_list = (data.select(self.cat_cols)
                         .distinct()
                         .orderBy(self.cat_cols)
                         ).rdd.map(lambda x: [xi for xi in x]).collect()
        return category_list

    @staticmethod
    def extract_feature_element(idx, feature):
        @F.pandas_udf(returnType=T.DoubleType())
        def _extract(feature):
            return feature[idx]

        return _extract(feature)

    def evaluate_cluster_plot(self,
                              df: DataFrame,
                              pca_k: int,
                              out_path: str,
                              out_filename: str = "clustering_fig",
                              seg_col: str = "segmentation",
                              figsize: Tuple[int] = (12, 10),
                              cmap_type: str = "Set1",
                              cmap_min: float = 0.,
                              cmap_max: float = 0.5,
                              sample: Optional[int] = None):
        """
        TODO: Not finished but this method will produce some evaluation plots for each cluster.
        """
        cmap = plt.cm.get_cmap(cmap_type)
        new_cmap = self.truncate_colormap(cmap, cmap_min, cmap_max)

        out_root_path = self.python_path(out_path)
        self.make_dir(out_root_path)

        for i, demog_groups in enumerate(self.category_list):
            demog_groups_str = str(demog_groups)
            demog_cat_filters = self._get_category_filters(demog_groups)
            df_temp = (df.filter(demog_cat_filters)
                       .select([f"pca_{k}" for k in range(pca_k)] + [seg_col])
                       )
            if sample:
                # sample is int limit of random Dataframe order.
                df_temp = df_temp.orderBy(F.rand()).limit(sample)

            df_ = df_temp.toPandas()

            fig, ax = plt.subplots()
            fig.set_size_inches(figsize)
            predictions = df_[seg_col].unique()

            scatter = ax.scatter(df_.loc[:, "pca_1"], df_.loc[:, "pca_2"], s=4, c=df_.loc[:, "prediction"], alpha=0.8,
                                 cmap=new_cmap)

            # produce a legend with the unique colors from the scatter
            legend1 = ax.legend(*scatter.legend_elements(num=predictions),
                                loc="upper right", title="Classes")
            ax.add_artist(legend1)
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")

            cluster_out_png_path = f"{out_root_path}/{out_filename}_{demog_groups_str}.png"
            print(f"{i}: {cluster_out_png_path}")
            fig.savefig(cluster_out_png_path)
            plt.close()

    @staticmethod
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    @staticmethod
    def python_path(path: str):
        return path.replace('dbfs:/', '/dbfs/')

    @staticmethod
    def make_dir(path: str):
        if not os.path.exists(path):
            os.mkdir(path)
