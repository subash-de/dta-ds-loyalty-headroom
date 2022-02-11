from pyspark.sql import functions as F, DataFrame
from typing import Optional, Union, Iterable, Dict, List
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
            exclude_items: Dict[str, str] = {"l3_id": ["MM14"]}
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

        # Remove items from transaction list, e.g. BWS items
        trx_line = self.remove_items(trx_line)

        if cust_seg is not None:
            # Only keep customers in segmentations
            trx_line = trx_line.join(cust_seg.select(self.user_key).distinct(), on=self.user_key)

        cust_lx_trx = self.get_customer_transactions(trx_line, lu_article)
        cust_lx_trx_metrics = self.get_transaction_metrics(cust_lx_trx)

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
                                         "basket_id", "sales_amt").filter(trx_line.SALES_AMT > 0.04)
                                 )
        # Find article ids of specific LX items
        lx_all = lu_article.filter(lu_article[f"{self.lx}_id"].isin(list(self.lx_ids)))
        # Find Customer Transactions who have purchased specific LX items
        customer_lx_transactions = customer_transactions.join(lx_all, ["article_id"])
        return customer_lx_transactions

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
                                     .agg(F.count(f"{self.lx}_name").alias("number_of_transactions"),
                                          F.sum("sales_amt").alias("total_spend"),
                                          F.count("article_id").alias("items"),
                                          F.countDistinct("basket_id").alias("visits"),
                                          (F.sum("sales_amt") / F.count("article_id")).alias("spend_per_item")
                                          )
                                     )

        # Find the sum of transactions and number of transactions
        customer_lx_trans_sum = (customer_lx_transactions
                                 .filter(F.col(self.user_key).isNotNull())
                                 .groupby([self.user_key])
                                 .agg(F.count(f"{self.lx}_name").alias("total_number_of_transactions"),
                                      F.sum("sales_amt").alias("total_spend_amount"),
                                      F.countDistinct("basket_id").alias("total_visits"),
                                      F.count("article_id").alias("total_items")
                                      )
                                 )

        # add number of baskets
        customer_lx_trans_count_basket = (customer_lx_transactions
                                          .filter(F.col(self.user_key).isNotNull())
                                          .groupby([self.user_key])
                                          .agg(F.countDistinct("basket_id").alias("sum_baskets"))
                                          )

        customer_lx_trans_count = (customer_lx_transactions
                                   .filter(F.col(self.user_key).isNotNull())
                                   .groupby([self.user_key])
                                   .agg(F.countDistinct(f"{self.lx}_name").alias(f"count_of_{self.lx}"))
                                   )

        customer_lx_trans_baskets = (customer_lx_transactions
                                     .filter(F.col(self.user_key).isNotNull())
                                     .groupby([self.user_key, f"{self.lx}_id", "basket_id"])
                                     .agg(F.sum("sales_amt").alias("total_spend_basket"),
                                          F.count("basket_id").alias("items_per_basket"))
                                     .groupby([self.user_key, f"{self.lx}_id"])
                                     .agg(F.mean("total_spend_basket").alias("average_basket_value"),
                                          F.expr('percentile_approx(total_spend_basket, 0.5)').alias("median_basket_value"),
                                          F.max("total_spend_basket").alias("max_basket_value"),
                                          F.mean("items_per_basket").alias("average_items_per_basket"),
                                          )
                                     )

        customer_lx_trans_baskets_full = (customer_lx_transactions
                                     .filter(F.col(self.user_key).isNotNull())
                                     .groupby([self.user_key, "basket_id"])
                                     .agg(F.sum("sales_amt").alias("total_spend_basket_full"),
                                          F.count("basket_id").alias("items_per_basket_full"))
                                     .groupby([self.user_key])
                                     .agg(F.mean("total_spend_basket_full").alias("average_basket_value_full"),
                                          F.expr('percentile_approx(total_spend_basket_full, 0.5)')
                                          .alias("median_basket_full_value"),
                                          F.max("total_spend_basket_full").alias("max_basket_full_value"),
                                          F.mean("items_per_basket_full").alias("average_items_per_basket_full"),
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
