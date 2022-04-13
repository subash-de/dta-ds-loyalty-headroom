from pyspark.sql import functions as F, DataFrame, Column, Window as W, types as T
from typing import Optional, Union, Iterable, Dict, List, Tuple
from datetime import datetime


# from great_expectations.dataset.sparkdf_dataset import SparkDFDataset

class BaseManager(object):

    @staticmethod
    def _strptime(date: Union[str, int], date_format: str):
        return datetime.strptime(str(date), date_format)

    @staticmethod
    def _add_date(df: DataFrame):
        return df.withColumn("date", F.col("year") * 10000 + F.col("month") * 100 + F.col("day"))

    @staticmethod
    def get_common_filters():
        common_filters = ((F.col("l2_id") != "FOPA") &
                          (F.col("trans_line_type") == "S") &
                          # Wrappers, Bags For Life, GIFT VOUCHER, VARIOUS - NO STAFF DISCOUNT
                          (~F.col("l4_id").isin("F98A", "F88A", "T46", "T63")) &
                          (F.col("area_name") != "PETROL STATIONS") &
                          (F.col("sales_amt") >= 0.1) &
                          (~F.col("l3_id").isin("40", "41")) &  # OTHER (NON MERCH), OTHER (CLOSED PRE 2014)
                          (F.col("trans_line_type") == "S") &
                          (F.col("cust_id").isNotNull())
                          )
        return common_filters

    @staticmethod
    def remove_christmas_transactions(df: DataFrame,
                                      christmas_range: Tuple[str] = ("1218", "0101")
                                      ) -> DataFrame:
        """
        Method to remove christmas period from data. Christmas is an atypical trading period.
        """

        if not "date" in df.columns:
            df = self._add_date(df)

        df_mmdd = (df
                   .withColumn("date_mmdd",
                               F.substring(F.col("date").cast(T.StringType()), 5, 4)
                               .cast(T.IntegerType()))
                   )

        christmas_start = int(christmas_range[0])
        christmas_end = int(christmas_range[1])

        return (df_mmdd
                .filter(~((F.col("date_mmdd") >= christmas_start) &
                          (F.col("date_mmdd") <= christmas_end)))
                .drop("date_mmdd")
                )


    @staticmethod
    def get_expr_agg(col: str,
                     pct_list: Tuple[float] = (50, 75, 85, 90, 100)
                     ) -> List[Column]:
        """
        method to fetch the statistics at defined percentile levels + the mean
        """
        out_expr = []
        #[F.mean(col).cast(T.DoubleType()).alias(f"mean_{col}")]
        for p in pct_list:
            pct = float(p/100.)
            out_expr.append(F.expr(f"percentile_approx({col}, {pct})")
                            .cast(T.DoubleType())
                            .alias(f"{p}percentile_{col}")
                            )
        return out_expr


class TransactionsManager(BaseManager):
    def __init__(
            self,
            start_date: str,
            end_date: str,
            l1_ids: list = ("GM"),
            lx: str = "l2",
            lx_ids: Iterable = ("01", "02", "03", "04", "05", "07"),
            user_key: str = "cust_id",
            date_format: Optional[str] = "%Y%m%d",
            # In store purchases only
            channels: List[str] = ["POS"],
            # No BWS, {"lx_id": [list, of, products, at lx, level]}
            exclude_items: Dict[str, str] = {"l3_id": ["MM14"]},
            window_days: Optional[int] = None,
            christmas_remove_range: Optional[Tuple[str]] = ("1218", "0101")
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.l1_ids = l1_ids
        self.lx = lx
        self.lx_ids = lx_ids
        self.user_key = user_key
        self.date_format = date_format
        self.channels = channels
        self.exclude_items = exclude_items
        self.window_days = window_days
        self.christmas_remove_range = christmas_remove_range

    def get(self,
            trx_line: DataFrame,
            lu_article: DataFrame,
            cust_seg: Optional[DataFrame] = None,
            ) -> DataFrame:
        """
        Entry method to run TransactionsManager
        """
        trx_line = (
            self._add_date(trx_line)
                .filter(F.col("date") >= self.start_date)
                .filter(F.col("date") <= self.end_date)
                .filter(F.col("PURCHASE_CHANNEL").isin(self.channels))
                .filter(F.col("l1_id").isin(list(self.l1_ids)))
                .filter(self.get_common_filters())
        )

        if self.christmas_remove_range is not None:
            trx_line = self.remove_christmas_transactions(trx_line,
                                                          christmas_range=self.christmas_remove_range)

        # Remove items from transaction list, e.g. BWS items
        trx_line = self.remove_items(trx_line)

        if cust_seg is not None:
            # Only keep customers in segmentations
            trx_line = trx_line.join(cust_seg.select(self.user_key).distinct(), on=self.user_key)

        cust_lx_trx = self.get_customer_transactions(trx_line, lu_article)
        cust_lx_trx_metrics = self.get_transaction_metrics(cust_lx_trx)

        if self.window_days is not None:
            trx_timespan = self.add_timespan_spend(cust_lx_trx)
            cust_lx_trx_metrics = (cust_lx_trx_metrics
                                   .join(trx_timespan, on=self.user_key, how="left")
                                   .fillna(0)
                                   )

        if cust_seg is not None:
            # Join on segmentation columns
            cust_lx_trx_metrics = cust_lx_trx_metrics.join(cust_seg
                                                           .dropDuplicates(subset=[self.user_key]),
                                                           on=self.user_key)

        return cust_lx_trx_metrics

    def remove_items(self, trx_data):
        """
        Method for removing items from the transaction table. required exclude_items input dictionary.
        """
        for (k, v) in self.exclude_items.items():
            trx_data = trx_data.filter(~(F.col(k).isin(v)))
        return trx_data

    def get_customer_transactions(self,
                                  trx_line: DataFrame,
                                  lu_article: DataFrame
                                  ) -> DataFrame:
        """
        Method to get customer level transactions
        """
        # Get LX hierarchy products
        customer_transactions = (trx_line
                                 .select(self.user_key, "cust_age", "cust_gender", "article_id",
                                         "basket_id", "sales_amt", "date")
                                 .filter(F.col("SALES_AMT") > 0.5)
                                 )
        # Find article ids of specific LX items
        lx_all = (lu_article
                  .filter(lu_article[f"{self.lx}_id"].isin(list(self.lx_ids)))
                  .select(["article_id"] +
                          [f"l{i}_id" for i in range(1, 7)] +
                          [f"l{i}_name" for i in range(1, 7)]
                          )
                  )
        # Find Customer Transactions who have purchased specific LX items
        customer_lx_transactions = customer_transactions.join(lx_all, ["article_id"])
        return customer_lx_transactions

    def add_timespan_spend(self,
                           cust_lx_trx: DataFrame
                           ) -> DataFrame:

        @F.udf(T.IntegerType())
        def days_back(date):
            days_diff = (datetime.strptime(str(self.end_date), self.date_format) -
                         datetime.strptime(str(date), self.date_format)).days
            return days_diff

        WinSpan = (W
                   .partitionBy(F.col("cust_id"))
                   .orderBy(F.col("day_diff").cast('long'))
                   .rangeBetween(-self.window_days, 0)
                   )

        trx_timespan = (cust_lx_trx
                        .withColumn("day_diff", days_back("date"))
                        .withColumn("spend_timespan", F.sum('sales_amt').over(WinSpan))
                        .groupby("cust_id")
                        .agg(*self.get_expr_agg("spend_timespan")
                             )
                        )
        return trx_timespan

    def get_transaction_metrics(self,
                                customer_lx_transactions: DataFrame
                                ) -> DataFrame:
        """
        Extract a series of metrics/stats from the customer level transaction table derived from the
        `get_customer_transactions` method.
        This includes:
        - number_of_transactions (count per lx & total over all lx)
        - total_spend (sum per lx & total over all lx)
        - items (count per lx & total over all lx)
        - visits (count per lx & total over all lx)
        - spend_per_item (mean per lx & total over all lx)
        - sum_baskets (count per lx & total over all lx)
        """

        # Find number of transactions per customer per l2 category
        customer_lx_trans_grouped = (customer_lx_transactions
                                     .filter(F.col(self.user_key).isNotNull())
                                     .groupby([self.user_key, f"{self.lx}_id"])
                                     #                                      .pivot(f"{self.lx}_id")
                                     .agg(F.count(f"{self.lx}_name")
                                          .cast(T.IntegerType()).alias("number_of_transactions"),
                                          F.sum("sales_amt")
                                          .cast(T.DoubleType()).alias("total_spend"),
                                          F.count("article_id")
                                          .cast(T.IntegerType()).alias("items"),
                                          F.countDistinct("basket_id")
                                          .cast(T.IntegerType()).alias("visits"),
                                          (F.sum("sales_amt") / F.count("article_id"))
                                          .cast(T.DoubleType()).alias("spend_per_item")
                                          )
                                     )

        # Find the sum of transactions and number of transactions
        customer_lx_trans_sum = (customer_lx_transactions
                                 .filter(F.col(self.user_key).isNotNull())
                                 .groupby([self.user_key])
                                 .agg(F.count(f"{self.lx}_name").cast(T.IntegerType())
                                      .alias("total_number_of_transactions"),
                                      F.sum("sales_amt").cast(T.DoubleType()).alias("total_spend_amount"),
                                      F.countDistinct("basket_id").cast(T.IntegerType()).alias("total_visits"),
                                      F.count("article_id").cast(T.IntegerType()).alias("total_items")
                                      )
                                 )

        # add number of baskets
        customer_lx_trans_count_basket = (customer_lx_transactions
                                          .filter(F.col(self.user_key).isNotNull())
                                          .groupby([self.user_key])
                                          .agg(F.countDistinct("basket_id").cast(T.IntegerType()).alias("sum_baskets"))
                                          )

        customer_lx_trans_count = (customer_lx_transactions
                                   .filter(F.col(self.user_key).isNotNull())
                                   .groupby([self.user_key])
                                   .agg(F.countDistinct(f"{self.lx}_name").alias(f"count_of_{self.lx}"))
                                   )

        customer_lx_trans_baskets = (customer_lx_transactions
                                     .filter(F.col(self.user_key).isNotNull())
                                     .groupby([self.user_key, f"{self.lx}_id", "basket_id"])
                                     .agg(F.sum("sales_amt").cast(T.DoubleType()).alias("total_spend_basket"),
                                          F.count("basket_id").cast(T.IntegerType()).alias("items_per_basket"))
                                     .groupby([self.user_key, f"{self.lx}_id"])
                                     .agg(*self.get_expr_agg("total_spend_basket"),
                                          *self.get_expr_agg("items_per_basket"),
                                          )
                                     )

        customer_lx_trans_baskets_full = (customer_lx_transactions
                                     .filter(F.col(self.user_key).isNotNull())
                                     .groupby([self.user_key, "basket_id"])
                                     .agg(F.sum("sales_amt").cast(T.DoubleType()).alias("total_spend_basket_full"),
                                          F.count("basket_id").cast(T.DoubleType()).alias("items_per_basket_full"))
                                     .groupby([self.user_key])
                                     .agg(*self.get_expr_agg("total_spend_basket_full"),
                                          *self.get_expr_agg("items_per_basket_full"),
                                          )
                                     )

        customer_lx_trans_grouped_all = (customer_lx_trans_grouped
                                         .join(customer_lx_trans_baskets, on=[self.user_key, f"{self.lx}_id"])
                                         .join(customer_lx_trans_sum, on=[self.user_key])
                                         .join(customer_lx_trans_count, on=[self.user_key])
                                         .join(customer_lx_trans_count_basket, on=[self.user_key])
                                         .join(customer_lx_trans_baskets_full, on=[self.user_key])
                                         )
        return customer_lx_trans_grouped_all


class IdMappingManager(object):
    def get(self, sparks_account: DataFrame) -> DataFrame:
        """
        Return cust_id to account_id mapping
        """
        cust_acc_mapping = (sparks_account
                            .filter(F.col("registration_date").isNotNull())
                            .select("cust_id", "account_id")
                            .distinct()
                            )
        return cust_acc_mapping
