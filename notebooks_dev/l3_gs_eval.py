# Databricks notebook source
# MAGIC %run ../notebooks/bootstrap

# COMMAND ----------

import os
from functools import partial
import pandas as pd
from datetime import datetime
from customer_headroom.etl.build_dataset import TransactionsManager
from customer_headroom.etl.segmentation import (
    SegmentationDataManager,
    SegmentationManager,
)
from customer_headroom.modelling.data_process import DataProcessor
from customer_headroom.modelling.fit import build_recommender
from customer_headroom.modelling.predict import Predictor
from customer_headroom.evaluation.model_selection import Evaluator
from customer_headroom.allocation.allocator import Allocator
import offerallocationv2.utils.persist_utils as persist_utils
from dtaml.logging import get_logger
from cdsutils.io_utils import file_exists, save_object, load_object
from multiprocessing.pool import ThreadPool
import seaborn as sns
from datetime import datetime, timedelta
from pyspark.sql import functions as F, DataFrame, Column, Window as W, types as T
from pyspark.sql.window import Window

sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

# COMMAND ----------

seg_list = [{'campaign': 20231127, 'experian_hh_composition': 'Cat_U', 'segmentation': 1}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_U', 'segmentation': 0}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_00', 'segmentation': 4}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_00', 'segmentation': 1}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_00', 'segmentation': 5}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_00', 'segmentation': 0}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_05', 'segmentation': 5}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_U', 'segmentation': 5}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_05', 'segmentation': 3}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_00', 'segmentation': 3}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_01', 'segmentation': 4}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_05', 'segmentation': 0}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_01', 'segmentation': 3}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_00', 'segmentation': 6}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_07', 'segmentation': 0}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_01', 'segmentation': 0}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_05', 'segmentation': 2}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_U', 'segmentation': 2}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_03', 'segmentation': 1}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_03', 'segmentation': 4}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_05', 'segmentation': 1}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_10', 'segmentation': 0}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_U', 'segmentation': 3}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_01', 'segmentation': 5}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_03', 'segmentation': 3}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_02', 'segmentation': 5}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_02', 'segmentation': 3}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_02', 'segmentation': 2}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_02', 'segmentation': 0}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_01', 'segmentation': 1}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_00', 'segmentation': 2}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_03', 'segmentation': 0}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_01', 'segmentation': 2}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_U', 'segmentation': 4}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_03', 'segmentation': 2}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_08', 'segmentation': 5}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_08', 'segmentation': 3}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_04', 'segmentation': 0}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_02', 'segmentation': 4}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_04', 'segmentation': 2}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_08', 'segmentation': 0}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_05', 'segmentation': 4}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_06', 'segmentation': 0}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_02', 'segmentation': 6}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_08', 'segmentation': 2}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_01', 'segmentation': 6}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_04', 'segmentation': 6}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_03', 'segmentation': 5}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_08', 'segmentation': 4}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_09', 'segmentation': 0}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_04', 'segmentation': 1}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_02', 'segmentation': 1}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_11', 'segmentation': 0}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_04', 'segmentation': 3}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_08', 'segmentation': 1}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_04', 'segmentation': 5}, {'campaign': 20231127, 'experian_hh_composition': 'Cat_04', 'segmentation': 4}]

# COMMAND ----------

def get_best_params(seg_list,config):
  
  config_fr = config['fit_rec']
  config_pd = config["predict"]
  partitionByList = config_pd["partitionByList"]

  best_params = pd.DataFrame()

  for seg in seg_list:
    seg_ext = [f"({k}='{seg[k]}')" for k in partitionByList]
    ext_str = "_".join([str(seg[k]) for k in partitionByList ])
    
    param_name = (config_fr.param_name + "_{ext}").format(ext=ext_str)
    param_model = persist_utils.load_model(model_name=param_name)

    keys = list(param_model.keys())
    values = list(param_model.values())

    df = pd.DataFrame([values])
    df.columns = keys
    df['seg'] = str(ext_str)

    best_params = pd.concat([best_params, df], ignore_index=True)

  return best_params

def get_gs_data(seg_list, config):

  config_fr = config['fit_rec']
  config_pd = config["predict"]
  partitionByList = config_pd["partitionByList"]

  all_gs_data = pd.DataFrame()

  for seg in seg_list:
    seg_ext = [f"({k}='{seg[k]}')" for k in partitionByList]
    ext_str = "_".join([str(seg[k]) for k in partitionByList ])
    
    gs_name = (config_fr.gs_name + "_{ext}").format(ext=ext_str)
    gs_model = persist_utils.load_model(model_name=gs_name)

    gs_dict = gs_model.cv_results
    gs_df = pd.DataFrame(gs_dict)
    gs_df['seg'] = str(ext_str)

    all_gs_data = pd.concat([all_gs_data, gs_df], ignore_index=True)

  return all_gs_data

# COMMAND ----------

gs_data = get_best_params(seg_list, config)

# COMMAND ----------

columns = ['seg','n_factors','n_epochs','lr_all','reg_all']
gs_data_spark = spark.createDataFrame(gs_data[columns])
df = (
    gs_data_spark.groupBy(columns[1:])
    .count()
    .withColumn(
        "paramaters_config",
        F.concat(
            F.col("n_factors"),
            F.lit("_"),
            F.col("n_epochs"),
            F.lit("_"),
            F.col("lr_all"),
            F.lit("_"),
            F.col("reg_all"),
        ),
    )
    .orderBy('count')
)

# COMMAND ----------

df.display()

# COMMAND ----------


