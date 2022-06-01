# Databricks notebook source
from datetime import datetime, timedelta
from functools import reduce, partial
from pyspark.sql import DataFrame, functions as F, types as T
from pyspark.sql.window import Window
from pyspark.ml.feature import PCA as sparkPCA
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from sklearn.metrics import r2_score

import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

# COMMAND ----------

dbutils.fs.ls("dbfs:/mnt/centralds/offerallocation/headroom/development/v0.2/data/202112/v0.0.0/")

# COMMAND ----------

# DBTITLE 1,TCO
# Read in sparks Customers
sparks_customers = spark.sql(f"SELECT * from analytics_trans_prod.sparks_account ")

# Read in sparks segtco
segtco_history = spark.sql(f"SELECT * from customer_azbase_prod.segtco_history")
segtco_history_ = (segtco_history
                   .filter(F.col("yyyymmdd")==20211211)
                  )

sparks_tco = (sparks_customers.select("cust_id", "account_id")
              .join(segtco_history_, on="cust_id", how="left")
              .withColumn("TCO", F.when(F.col("cust_band_ch")=="1. Top", "1. Top")
                          .when(F.col("cust_band_ch")=="2. Core", "2. Core")
                          .when(F.col("cust_band_ch")=="3. Occ", "3. Occ")
                          .otherwise("7.Null")
                         )
              .select("cust_id", "account_id", "TCO")
              .distinct()
             )
display(sparks_tco)

# COMMAND ----------

data_path = "dbfs:/mnt/centralds/offerallocation/headroom/prod_run/v0.3/data/202112/v0.0.0/purch_hist"

data = spark.read.parquet(data_path)


seg_path = "dbfs:/mnt/centralds/offerallocation/headroom/prod_run/v0.3/data/202112/v0.0.0/cust_segment"

seg = spark.read.parquet(seg_path)


#  Out
temp_path = "dbfs:/mnt/centralds/offerallocation/headroom/prod_run/v0.3/plots/202112/v0.0.0/clustering/"
out_png = f"{temp_path}/out/"

display(seg)

# COMMAND ----------

display(seg.groupby("experian_hh_composition", "segmentation").count())

# COMMAND ----------




cmap = plt.cm.get_cmap('Set1')
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
  
new_cmap = truncate_colormap(cmap, 0.3, 0.5)

out_path = out_png.replace('dbfs:/', '/dbfs/')
if not os.path.exists(out_path):
  os.makedirs(out_path)


hhs = seg.select("experian_hh_composition").distinct().rdd.map(lambda x: x[0]).collect()

# COMMAND ----------


for i, hh in enumerate(hhs):
  df = (seg.filter(F.col("experian_hh_composition")==hh)
        .join(sparks_tco, on="cust_id", how="left")
        .join(data.select("cust_id", "total_spend_amount").distinct(), on="cust_id", how="left")
       )
  df_ = df.orderBy(F.rand()).limit(10000).toPandas()
  
  
  fig, ax = plt.subplots()
  fig.set_size_inches((12, 10))
  segmentations = df_.segmentation.unique()


  num_cols = ["age_to_use", "gender_F", "gender_M", "FD_spend", "GM_spend", "FD_items",  "GM_items", "FD_baskets", "GM_baskets" ]
  pca = PCA(n_components=2)
  pca.fit(df_.loc[:, num_cols])
  pcas = pca.transform(df_.loc[:, num_cols])

  scatter = ax.scatter(pcas[:, 0], pcas[:, 1], s=4, c=df_.loc[:, "segmentation"], alpha=0.8, cmap=new_cmap)

  # produce a legend with the unique colors from the scatter
  legend1 = ax.legend(*scatter.legend_elements(num=segmentations),
                      loc="upper right", title="Segmentation")
  ax.add_artist(legend1)
  ax.set_xlabel("PCA 0")
  ax.set_ylabel("PCA 1")

  cluster_out_png_path = f"{out_path}/clustering_test_{hh}.png"
  print(f"{i}: {cluster_out_png_path}")
  fig.savefig(cluster_out_png_path)
  plt.close()
  
  
  n_seg = len(segmentations)
  fig, axes = plt.subplots(7,n_seg)
  fig.set_size_inches((10*n_seg, 32))
  for s in segmentations:
    axes[0, s].hist(df_[df_.segmentation==s].age_to_use, bins=10)
    axes[0, s].set_xlabel("age")
    axes[0, s].set_title(f"Age. Segment: {s}")
    axes[1, s].hist(df_[df_.segmentation==s].FD_spend, bins=10)
    axes[1, s].set_xlabel("FD_spend")
    axes[1, s].set_title(f"FD_spend. Segment: {s}")
    axes[2, s].hist(df_[df_.segmentation==s].FD_items, bins=10)
    axes[2, s].set_xlabel("FD_items")
    axes[2, s].set_title(f"FD_items. Segment: {s}")
    axes[3, s].hist(df_[df_.segmentation==s].FD_baskets, bins=10)
    axes[3, s].set_xlabel("FD_baskets")
    axes[3, s].set_title(f"FD_baskets. Segment: {s}")
    axes[4, s].hist(df_[df_.segmentation==s].total_spend_amount, bins=100)
    axes[4, s].set_xlabel("FD_baskets")
    axes[4, s].set_title(f"total_spend_amount. Segment: {s}")
    df_[df_.segmentation==s].TCO.value_counts().sort_index().plot(kind='bar', ax=axes[5, s])
    axes[5, s].set_title(f"TCO. Segment: {s}")
    df_[df_.segmentation==s].gender.value_counts().sort_index().plot(kind='pie', ax=axes[6, s])
    axes[6, s].set_title(f"Gender. Segment: {s}")
    plt.tight_layout()
    
  stats_cluster_out_png_path = f"{out_path}/stats_clustering_test_{hh}.png"
  print(f"{i}: {stats_cluster_out_png_path}")
  fig.savefig(stats_cluster_out_png_path)
  plt.close()

# COMMAND ----------


