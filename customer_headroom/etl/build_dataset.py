from pyspark.sql import functions as F, DataFrame, Column, Window as W, types as T
from typing import Optional, Union, Iterable, Dict, List, Tuple
from datetime import datetime, timedelta


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

    def remove_christmas_transactions(self,
                                      df: DataFrame,
                                      christmas_range: Tuple[str] = ("1218", "0101")
                                      ) -> DataFrame:
        """
        Method to remove christmas period from data. Christmas is an atypical trading period.
        """

        if ~("date" in df.columns):
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
        out_expr = [F.mean(col).cast(T.DoubleType()).alias(f"average_{col}")]
        for p in pct_list:
            pct = float(p / 100.)
            out_expr.append(F.expr(f"percentile_approx({col}, {pct})")
                            .cast(T.DoubleType())
                            .alias(f"{p}percentile_{col}")
                            )
        return out_expr


class TransactionsManager(BaseManager):
    def __init__(
            self,
            etl_date: str,
            lookback_days: str,
            l1_ids: list = ("GM"),
            lx: str = "l2",
            lx_ids: Iterable = ("01", "02", "03", "04", "05", "07"),
            user_key: str = "cust_id",
            aggregation_level: str = "basket",
            date_format: Optional[str] = "%Y%m%d",
            # In store purchases only
            channels: List[str] = ["POS"],
            # No BWS, {"lx_id": [list, of, products, at lx, level]}
            exclude_items: Dict[str, str] = {"l3_id": ["MM14"]},
            window_days: Optional[int] = None,
            christmas_remove_range: Optional[Tuple[str]] = ("1218", "0101"),
            time_window_length: Optional[int] = None,
            
    ):
        self.etl_date = etl_date
        self.lookback_days = lookback_days
        self.lookback_date = int((datetime.strptime(str(etl_date), date_format) -
                                  timedelta(days=lookback_days)).strftime(date_format))
        self.l1_ids = l1_ids
        self.lx = lx
        self.lx_ids = lx_ids
        self.user_key = user_key
        self.date_format = date_format
        self.channels = channels
        self.exclude_items = exclude_items
        self.window_days = window_days
        self.christmas_remove_range = christmas_remove_range
        self.time_window_length = time_window_length
        self.aggregation_level = aggregation_level

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
                .filter(F.col("date") <= self.etl_date)
                .filter(F.col("date") >= self.lookback_date)
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

        # Add a time window column to groupby 
        if self.time_window_length is not None: 
            time_window_ind_df = self.add_time_window_ind(cust_lx_trx = cust_lx_trx)
            cust_lx_trx = (cust_lx_trx
                                    .join(time_window_ind_df, on = "date", how = 'left'))


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
        if self.exclude_items is None or self.exclude_items == {} or self.exclude_items == "None": 
          return trx_data
        else:
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
                  .filter(lu_article["l2_id"].isin(list(self.lx_ids)))
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
            days_diff = (datetime.strptime(str(self.lookback_date), self.date_format) -
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

    # def add_time_window_ind(self, 
    #                         cust_lx_trx: DataFrame
    #                         )-> DataFrame:
    #     """Calcuate the time window, difference in days between the transaction date, and etl date divided by the window length

    #     Args:
    #         cust_lx_trx (DataFrame): _description_

    #     Returns:
    #         DataFrame: _description_
    #     """
    #     @F.udf(T.IntegerType())
    #     def time_window_back(date):
    #         days_diff = (datetime.strptime(str(self.etl_date), self.date_format) - 
    #                     datetime.strptime(str(date), self.date_format)).days // self.time_window_length
    #         return days_diff 
        
    #     trx_time_window = (cust_lx_trx
    #                     .select("date")
    #                     .withColumn("time_window_ind", time_window_back("date")))

    #     return trx_time_window

    def add_time_window_ind(self, cust_lx_trx) -> DataFrame:
        @F.udf(T.IntegerType())
        def time_window_back(date):
            days_diff = (datetime.strptime(str(self.etl_date), self.date_format) - 
                            datetime.strptime(str(date), self.date_format)).days // self.time_window_length
            return days_diff 

        trx_time_window = (
            cust_lx_trx
            .select("date")
            .distinct()
            .withColumn("time_window_ind", time_window_back("date"))
            )  
        return trx_time_window


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

        customer_lx_transactions.cache()

        # Find number of transactions per customer per l2 category
        customer_lx_trans_grouped = (customer_lx_transactions
                                     .filter(F.col(self.user_key).isNotNull())
                                     .groupby([self.user_key, f"{self.lx}_id"])
                                    # .groupby([self.user_key])
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


        customer_lx_trans_time_window= (customer_lx_transactions
                                     .filter(F.col(self.user_key).isNotNull())
                                     .groupby([self.user_key, f"{self.lx}_id", "time_window_ind"])
                                     .agg(F.sum("sales_amt").cast(T.DoubleType()).alias("total_spend_time_window"),
                                          F.countDistinct("basket_id").cast(T.IntegerType()).alias("basket_per_time_window"))
                                     .groupby([self.user_key, f"{self.lx}_id"])
                                     .agg(*self.get_expr_agg("total_spend_time_window"),
                                          *self.get_expr_agg("basket_per_time_window"),
                                          )
                                     )

        # max time window basket 
        customer_lx_trans_weekly_max_basket = (
            customer_lx_transactions
            .filter(F.col(self.user_key).isNotNull())
            .groupby([self.user_key, f"{self.lx}_id", "basket_id", "time_window_ind"])
            .agg(F.sum("sales_amt").cast(T.DoubleType()).alias("total_spend_basket"),
            F.count("basket_id").cast(T.IntegerType()).alias("items_per_basket"))
            .groupby([self.user_key, f"{self.lx}_id", "time_window_ind"])
            .agg(F.max("total_spend_basket").cast(T.DoubleType()).alias("time_window_max_spend_basket"))
            .groupby([self.user_key, f"{self.lx}_id"])
            .agg(*self.get_expr_agg("time_window_max_spend_basket"))
        )

        # need to do melt / unpivot, melt exist for newer version of pyspark >=3.4.0
        unpivotExpr = "stack(6, 'average_time_window_max_spend_basket', average_time_window_max_spend_basket, \
          '50percentile_time_window_max_spend_basket', 50percentile_time_window_max_spend_basket, \
            '75percentile_time_window_max_spend_basket', 75percentile_time_window_max_spend_basket, \
            '85percentile_time_window_max_spend_basket', 85percentile_time_window_max_spend_basket, \
            '90percentile_time_window_max_spend_basket', 90percentile_time_window_max_spend_basket, \
            '100percentile_time_window_max_spend_basket', 100percentile_time_window_max_spend_basket ) as (l2_id,weekly_max_basket_percentile)"

        # find the percentile of the time window (weekly) max basket value
        customer_weekly_max_transaction = ( 
          customer_lx_transactions
          .select(self.user_key, "basket_id", "time_window_ind", "sales_amt")
          # find the basket amount 
          .groupby(self.user_key, "basket_id", "time_window_ind")
          .agg(F.sum("sales_amt").cast(T.DoubleType()).alias("total_spend_basket"))
          # find the time window (weekly) max basket amount 
          .groupby(self.user_key, "time_window_ind")
          .agg(F.max("total_spend_basket").cast(T.DoubleType()).alias("time_window_max_spend_basket"))
          .groupby("cust_id")
          # calculate the percentile of max basket in a time window (weekly)
          .agg(*self.get_expr_agg("time_window_max_spend_basket")) 
          # To make the code work, instead of the l2 categories, will replace with percentile
          .select("cust_id", F.expr(unpivotExpr))
          .filter(~F.col("l2_id").isNull())
        )


        # calculate the l2 id of the nearest 85th percentile weekly max basket 
        percentile_spend = (  
          customer_lx_transactions
          .select("cust_id", "basket_id", "time_window_ind", "sales_amt")
          # find the basket amount 
          .groupby("cust_id", "basket_id", "time_window_ind")
          .agg(F.sum("sales_amt").cast(T.DoubleType()).alias("total_spend_basket"))
          # find the weeky max basket amount 
          .groupby("cust_id", "time_window_ind")
          .agg(F.max("total_spend_basket").cast(T.DoubleType()).alias("time_window_max_spend_basket"))
          .groupby("cust_id")
          .agg(*self.get_expr_agg("time_window_max_spend_basket"))
          .select("cust_id", "85percentile_time_window_max_spend_basket")  ) # add to config ================
        
        # get the basket id that is closest to the 85th percentile
        max_basket_id = (
          customer_lx_transactions
          .select("cust_id", "basket_id", "time_window_ind", "sales_amt")
          # find the basket amount 
          .groupby("cust_id", "basket_id", "time_window_ind")
          .agg(F.sum("sales_amt").cast(T.DoubleType()).alias("total_spend_basket"))
          # find the weeky max basket amount 
          # .groupby("cust_id", "WEEK_ID")
          .withColumn('time_window_max_spend_basket', F.max("total_spend_basket").over(W.partitionBy("cust_id", "time_window_ind") ))
          .where(F.col("total_spend_basket") == F.col("time_window_max_spend_basket"))
          # .withColumn("percentile", F.lit(34.94))
          .join(percentile_spend, how = 'left', on = "cust_id")
          .where(F.col("time_window_max_spend_basket") >= F.col("85percentile_time_window_max_spend_basket"))
          # .orderBy("time_window_max_spend_basket")
          .withColumn("row", F.row_number().over(W.partitionBy("cust_id").orderBy(F.col("time_window_max_spend_basket"))))
          .filter(F.col("row") == 1)
        )

        # find the l2 id spend for the given basket
        customer_lx_basket_spend = (
          customer_lx_transactions
          .join(max_basket_id.select("cust_id", "basket_id"), how = 'inner', on = ["cust_id", "basket_id"])
          .groupby("cust_id", f"{self.lx}_id").agg(F.sum("sales_amt").cast(T.DoubleType()).alias(f"{self.lx}_id_total_spend_basket"))
        )

        # ================================================
        # calculate the weekly total spend 

        #new logic for build dataset 
        lx_id_time_window_spend = (
            customer_lx_transactions.select(
                self.user_key, f"{self.lx}_id", "time_window_ind", "sales_amt"
            )
            # find the spend in time window for each lx_id
            .groupby(self.user_key, f"{self.lx}_id", "time_window_ind")
            .agg(F.sum("sales_amt").cast(T.DoubleType()).alias("total_spend_time_window"))
            # get 85.00 percentile spend in time window for each lx_id
            .groupby(self.user_key, f"{self.lx}_id")
            .agg(*self.get_expr_agg("total_spend_time_window"))
            .select(self.user_key, f"{self.lx}_id", "85percentile_total_spend_time_window")
            .withColumn(f"{self.lx}_id_total_time_window_spend", F.col('85percentile_total_spend_time_window'))
        )

        percentile_spend_time_window = (  
          customer_lx_transactions
          .select("cust_id", "time_window_ind", "sales_amt")
          # find the spend in time window 
          .groupby("cust_id", "time_window_ind")
          .agg(F.sum("sales_amt").cast(T.DoubleType()).alias("total_spend_time_window"))
          # find the weeky max basket amount 
          # .groupby("cust_id", "time_window_ind")
          # .agg(F.max("total_spend_basket").cast(T.DoubleType()).alias("time_window_max_spend_basket"))
          .groupby("cust_id")
          .agg(*self.get_expr_agg("total_spend_time_window"))
          .select("cust_id", "85percentile_total_spend_time_window")  ) # add to config ================


        # get the time window ind id that is closest to the 85th percentile
        time_window_ind_id = (
          customer_lx_transactions
          .select("cust_id", "time_window_ind", "sales_amt")
          # find the basket amount 
          .groupby("cust_id", "time_window_ind")
          .agg(F.sum("sales_amt").cast(T.DoubleType()).alias("total_spend_time_window"))
          # find the weeky max basket amount 
          # .groupby("cust_id", "WEEK_ID")
          # .withColumn('time_window_max_spend_basket', F.max("total_spend_basket").over(W.partitionBy("cust_id", "time_window_ind") ))
          # .where(F.col("total_spend_basket") == F.col("time_window_max_spend_basket"))
          # .withColumn("percentile", F.lit(34.94))
          .join(percentile_spend_time_window, how = 'left', on = "cust_id")
          .where(F.col("total_spend_time_window") >= F.col("85percentile_total_spend_time_window"))
          # .orderBy("time_window_max_spend_basket")
          .withColumn("row", F.row_number().over(W.partitionBy("cust_id").orderBy(F.col("total_spend_time_window"))))
          .filter(F.col("row") == 1)
        )

        time_window_ind_id.cache()

        # find the l2 id spend for the given time window
        customer_lx_time_window_spend = (
          customer_lx_transactions
          .join(time_window_ind_id.select("cust_id", "time_window_ind"), how = 'inner', on = ["cust_id", "time_window_ind"])
          .groupby("cust_id", f"{self.lx}_id").agg(F.sum("sales_amt").cast(T.DoubleType()).alias(f"{self.lx}_id_total_time_window_spend"))
        )
        # ================================================


        # ================================================
        # calculate the count distinct basket and time window. 
        # noticed that for accumulator, there are people who had very few baskets 
        customer_overall_count = (
          customer_lx_transactions
          .filter(F.col(self.user_key).isNotNull())
          .groupby(self.user_key)
          .agg(F.countDistinct("basket_id").cast(T.IntegerType()).alias("count_user_basket"),
               F.countDistinct("time_window_ind").cast(T.IntegerType()).alias("count_user_time_window")
          )
        )
        # ================================================
        if self.aggregation_level == 'basket':
          customer_lx_trans_grouped_all = (customer_lx_trans_grouped
                                        #  .join(customer_lx_basket_spend, on = [self.user_key, f"{self.lx}_id"])
                                        # .join(df, on = [self.user_key, f"{self.lx}_id"])
                                         .join(customer_lx_time_window_spend, on = [self.user_key, f"{self.lx}_id"])
                                         .join(customer_overall_count, on = [self.user_key])
                                        #  .join(customer_weekly_max_transaction, on = [self.user_key], how = "outer")
                                        #  .join(customer_lx_trans_baskets, on=[self.user_key, f"{self.lx}_id"])
                                        #  .join(customer_lx_trans_sum, on=[self.user_key])
                                        #  .join(customer_lx_trans_count, on=[self.user_key])
                                        #  .join(customer_lx_trans_count_basket, on=[self.user_key])
                                        #  .join(customer_lx_trans_baskets_full, on=[self.user_key])
                                        #  .join(customer_lx_trans_time_window, on = [self.user_key, f"{self.lx}_id"])
                                        #  .join(customer_lx_trans_weekly_max_basket, on = [self.user_key, f"{self.lx}_id"])
                                         )
        
        else:
          customer_lx_trans_grouped_all = (customer_lx_trans_grouped
                                          #  .join(customer_lx_basket_spend, on = [self.user_key, f"{self.lx}_id"])
                                          .join(lx_id_time_window_spend, on = [self.user_key, f"{self.lx}_id"])
                                          # .join(customer_lx_time_window_spend, on = [self.user_key, f"{self.lx}_id"])
                                          .join(customer_overall_count, on = [self.user_key])
                                          #  .join(customer_weekly_max_transaction, on = [self.user_key], how = "outer")
                                          #  .join(customer_lx_trans_baskets, on=[self.user_key, f"{self.lx}_id"])
                                          #  .join(customer_lx_trans_sum, on=[self.user_key])
                                          #  .join(customer_lx_trans_count, on=[self.user_key])
                                          #  .join(customer_lx_trans_count_basket, on=[self.user_key])
                                          #  .join(customer_lx_trans_baskets_full, on=[self.user_key])
                                          #  .join(customer_lx_trans_time_window, on = [self.user_key, f"{self.lx}_id"])
                                          #  .join(customer_lx_trans_weekly_max_basket, on = [self.user_key, f"{self.lx}_id"])
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
