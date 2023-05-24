# Databricks notebook source


# COMMAND ----------

# MAGIC %sql select case
# MAGIC         when customer_segment = 1 then 'grab and goers'
# MAGIC         when customer_segment = 2 then 'magic seekers'
# MAGIC         when customer_segment = 3 then 'basket builders'
# MAGIC         when customer_segment = 4 then 'easy eaters'
# MAGIC         when customer_segment = 5 then 'savvy savers'
# MAGIC         when customer_segment = 6 then 'social shoppers'
# MAGIC         else 'Err' end as segment_desc, account_id
# MAGIC from
# MAGIC fci_azlab_dev.sg_segmentation_cust_base_scores_all
# MAGIC where year_end = 20230401

# COMMAND ----------


