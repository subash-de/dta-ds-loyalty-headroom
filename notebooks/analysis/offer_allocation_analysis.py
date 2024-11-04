# Databricks notebook source
# MAGIC %run ../setup/bootstrap

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.functions import col, weekofyear, to_date, expr, datediff
from pyspark.sql.window import Window

# COMMAND ----------

opt_in_cust = spark.sql("""
                    select
                        distinct
                        cust_id,
                        account_id,
                        staff_ind,
                        wcs_id,
                        sparks_joined_date
                    from
                        campaign_analyse_prod.campaign_eligibility_p_tbl
                    where
                        channel = 'Email'
                        and country = 'UK'
                        and type = 'Sparks'
                        and marketing_status = 'Opt-in Active'
                    """)

# COMMAND ----------

from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger
from pyspark.sql import Window as W
from pyspark.sql import functions as F

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.allocation.allocator import Allocator
from customer_headroom.utils import tmo_utils

sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

# COMMAND ----------

config_al = config['allocation']

# COMMAND ----------

def get_campaign(campaign, etl_date):
    if (campaign == "{campaign}") or (campaign == ""):
        campaign = get_date(etl_date)
    return campaign
campaign = get_campaign(config.dates.upcoming_campaign, config.dates.etl_date)

# COMMAND ----------

# Offer allocation table
test_cells_tbl_name = persist_utils.get_table_name(
    factory_database=config.factory_database,
    lab_database=config.lab_database,
    table_prefix=config_al.full_export_tbl.prefix,
    sensitivity=config.sensitivity,
)
logger.info(f"""test_cells_tbl_name: {test_cells_tbl_name}""")

offers_export = persist_utils.read_table(
    table_name=test_cells_tbl_name, where=f"campaign={campaign}"
)

# COMMAND ----------

offers_export = offers_export.filter(F.col("spend_plus_stretch") <= 270)

# COMMAND ----------

import numpy as np

# COMMAND ----------

offers_export_pandas = offers_export.toPandas()

# COMMAND ----------

offers_10 = ['£5 off when you spend £30 on M&S food in store',
 '£5 off when you spend £40 on M&S food in store',
 '£5 off when you spend £50 on M&S food in store',
 '£6 off when you spend £60 on M&S food in store',
 '£7 off when you spend £70 on M&S food in store',
 '£8 off when you spend £80 on M&S food in store',
 '£9 off when you spend £90 on M&S food in store',
 '£10 off when you spend £100 on M&S food in store',
 '£11 off when you spend £110 on M&S food in store',
 '£12 off when you spend £120 on M&S food in store',
 '£13 off when you spend £130 on M&S food in store',
 '£14 off when you spend £140 on M&S food in store',
 '£15 off when you spend £150 on M&S food in store',
 '£16 off when you spend £160 on M&S food in store',
 '£17 off when you spend £170 on M&S food in store',
 '£18 off when you spend £180 on M&S food in store',
 '£19 off when you spend £190 on M&S food in store',
 '£20 off when you spend £200 on M&S food in store',
 '£21 off when you spend £210 on M&S food in store',
 '£22 off when you spend £220 on M&S food in store',
 '£23 off when you spend £230 on M&S food in store',
 '£24 off when you spend £240 on M&S food in store',
 '£25 off when you spend £250 on M&S food in store',
 '£26 off when you spend £260 on M&S food in store',
 '£27 off when you spend £270 on M&S food in store']
conditions_10= [
    (offers_export_pandas['spend_plus_stretch'] >= 0) & (offers_export_pandas['spend_plus_stretch'] < 30),
    (offers_export_pandas['spend_plus_stretch'] >= 30) & (offers_export_pandas['spend_plus_stretch'] < 40),
    (offers_export_pandas['spend_plus_stretch'] >= 40) & (offers_export_pandas['spend_plus_stretch'] < 50),
    (offers_export_pandas['spend_plus_stretch'] >= 50) & (offers_export_pandas['spend_plus_stretch'] < 60),
    (offers_export_pandas['spend_plus_stretch'] >= 60) & (offers_export_pandas['spend_plus_stretch'] < 70),
    (offers_export_pandas['spend_plus_stretch'] >= 70) & (offers_export_pandas['spend_plus_stretch'] < 80),
    (offers_export_pandas['spend_plus_stretch'] >= 80) & (offers_export_pandas['spend_plus_stretch'] < 90),
    (offers_export_pandas['spend_plus_stretch'] >= 90) & (offers_export_pandas['spend_plus_stretch'] < 100),
    (offers_export_pandas['spend_plus_stretch'] >= 100) & (offers_export_pandas['spend_plus_stretch'] < 110),
    (offers_export_pandas['spend_plus_stretch'] >= 110) & (offers_export_pandas['spend_plus_stretch'] < 120),
    (offers_export_pandas['spend_plus_stretch'] >= 120) & (offers_export_pandas['spend_plus_stretch'] < 130),
    (offers_export_pandas['spend_plus_stretch'] >= 130) & (offers_export_pandas['spend_plus_stretch'] < 140),
    (offers_export_pandas['spend_plus_stretch'] >= 140) & (offers_export_pandas['spend_plus_stretch'] < 150),
    (offers_export_pandas['spend_plus_stretch'] >= 150) & (offers_export_pandas['spend_plus_stretch'] < 160),
    (offers_export_pandas['spend_plus_stretch'] >= 160) & (offers_export_pandas['spend_plus_stretch'] < 170),
    (offers_export_pandas['spend_plus_stretch'] >= 170) & (offers_export_pandas['spend_plus_stretch'] < 180),
    (offers_export_pandas['spend_plus_stretch'] >= 180) & (offers_export_pandas['spend_plus_stretch'] < 190),
    (offers_export_pandas['spend_plus_stretch'] >= 190) & (offers_export_pandas['spend_plus_stretch'] < 200),
    (offers_export_pandas['spend_plus_stretch'] >= 200) & (offers_export_pandas['spend_plus_stretch'] < 210),
    (offers_export_pandas['spend_plus_stretch'] >= 210) & (offers_export_pandas['spend_plus_stretch'] < 220),
    (offers_export_pandas['spend_plus_stretch'] >= 220) & (offers_export_pandas['spend_plus_stretch'] < 230),
    (offers_export_pandas['spend_plus_stretch'] >= 230) & (offers_export_pandas['spend_plus_stretch'] < 240),
    (offers_export_pandas['spend_plus_stretch'] >= 240) & (offers_export_pandas['spend_plus_stretch'] < 250),
    (offers_export_pandas['spend_plus_stretch'] >= 250) & (offers_export_pandas['spend_plus_stretch'] < 260),
    (offers_export_pandas['spend_plus_stretch'] >= 260) & (offers_export_pandas['spend_plus_stretch'] < 270)
]


# COMMAND ----------

conditions_5 = [
    (offers_export_pandas['spend_plus_stretch'] >= 0) & (offers_export_pandas['spend_plus_stretch'] < 30),
    (offers_export_pandas['spend_plus_stretch'] >= 30) & (offers_export_pandas['spend_plus_stretch'] < 35),
    (offers_export_pandas['spend_plus_stretch'] >= 35) & (offers_export_pandas['spend_plus_stretch'] < 40),
    (offers_export_pandas['spend_plus_stretch'] >= 40) & (offers_export_pandas['spend_plus_stretch'] < 45),
    (offers_export_pandas['spend_plus_stretch'] >= 45) & (offers_export_pandas['spend_plus_stretch'] < 50),
    (offers_export_pandas['spend_plus_stretch'] >= 50) & (offers_export_pandas['spend_plus_stretch'] < 55),
    (offers_export_pandas['spend_plus_stretch'] >= 55) & (offers_export_pandas['spend_plus_stretch'] < 60),
    (offers_export_pandas['spend_plus_stretch'] >= 60) & (offers_export_pandas['spend_plus_stretch'] < 65),
    (offers_export_pandas['spend_plus_stretch'] >= 65) & (offers_export_pandas['spend_plus_stretch'] < 70),
    (offers_export_pandas['spend_plus_stretch'] >= 70) & (offers_export_pandas['spend_plus_stretch'] < 75),
    (offers_export_pandas['spend_plus_stretch'] >= 75) & (offers_export_pandas['spend_plus_stretch'] < 80),
    (offers_export_pandas['spend_plus_stretch'] >= 80) & (offers_export_pandas['spend_plus_stretch'] < 85),
    (offers_export_pandas['spend_plus_stretch'] >= 85) & (offers_export_pandas['spend_plus_stretch'] < 90),
    (offers_export_pandas['spend_plus_stretch'] >= 90) & (offers_export_pandas['spend_plus_stretch'] < 95),
    (offers_export_pandas['spend_plus_stretch'] >= 95) & (offers_export_pandas['spend_plus_stretch'] < 100),
    (offers_export_pandas['spend_plus_stretch'] >= 100) & (offers_export_pandas['spend_plus_stretch'] < 105),
    (offers_export_pandas['spend_plus_stretch'] >= 105) & (offers_export_pandas['spend_plus_stretch'] < 110),
    (offers_export_pandas['spend_plus_stretch'] >= 110) & (offers_export_pandas['spend_plus_stretch'] < 115),
    (offers_export_pandas['spend_plus_stretch'] >= 115) & (offers_export_pandas['spend_plus_stretch'] < 120),
    (offers_export_pandas['spend_plus_stretch'] >= 120) & (offers_export_pandas['spend_plus_stretch'] < 125),
    (offers_export_pandas['spend_plus_stretch'] >= 125) & (offers_export_pandas['spend_plus_stretch'] < 130),
    (offers_export_pandas['spend_plus_stretch'] >= 130) & (offers_export_pandas['spend_plus_stretch'] < 135),
    (offers_export_pandas['spend_plus_stretch'] >= 135) & (offers_export_pandas['spend_plus_stretch'] < 140),
    (offers_export_pandas['spend_plus_stretch'] >= 140) & (offers_export_pandas['spend_plus_stretch'] < 145),
    (offers_export_pandas['spend_plus_stretch'] >= 145) & (offers_export_pandas['spend_plus_stretch'] < 150),
    (offers_export_pandas['spend_plus_stretch'] >= 150) & (offers_export_pandas['spend_plus_stretch'] < 155),
    (offers_export_pandas['spend_plus_stretch'] >= 155) & (offers_export_pandas['spend_plus_stretch'] < 160),
    (offers_export_pandas['spend_plus_stretch'] >= 160) & (offers_export_pandas['spend_plus_stretch'] < 165),
    (offers_export_pandas['spend_plus_stretch'] >= 165) & (offers_export_pandas['spend_plus_stretch'] < 170),
    (offers_export_pandas['spend_plus_stretch'] >= 170) & (offers_export_pandas['spend_plus_stretch'] < 175),
    (offers_export_pandas['spend_plus_stretch'] >= 175) & (offers_export_pandas['spend_plus_stretch'] < 180),
    (offers_export_pandas['spend_plus_stretch'] >= 180) & (offers_export_pandas['spend_plus_stretch'] < 185),
    (offers_export_pandas['spend_plus_stretch'] >= 185) & (offers_export_pandas['spend_plus_stretch'] < 190),
    (offers_export_pandas['spend_plus_stretch'] >= 190) & (offers_export_pandas['spend_plus_stretch'] < 195),
    (offers_export_pandas['spend_plus_stretch'] >= 195) & (offers_export_pandas['spend_plus_stretch'] < 200),
    (offers_export_pandas['spend_plus_stretch'] >= 200) & (offers_export_pandas['spend_plus_stretch'] < 205),
    (offers_export_pandas['spend_plus_stretch'] >= 205) & (offers_export_pandas['spend_plus_stretch'] < 210),
    (offers_export_pandas['spend_plus_stretch'] >= 210) & (offers_export_pandas['spend_plus_stretch'] < 215),
    (offers_export_pandas['spend_plus_stretch'] >= 215) & (offers_export_pandas['spend_plus_stretch'] < 220),
    (offers_export_pandas['spend_plus_stretch'] >= 220) & (offers_export_pandas['spend_plus_stretch'] < 225),
    (offers_export_pandas['spend_plus_stretch'] >= 225) & (offers_export_pandas['spend_plus_stretch'] < 230),
    (offers_export_pandas['spend_plus_stretch'] >= 230) & (offers_export_pandas['spend_plus_stretch'] < 235),
    (offers_export_pandas['spend_plus_stretch'] >= 235) & (offers_export_pandas['spend_plus_stretch'] < 240),
    (offers_export_pandas['spend_plus_stretch'] >= 240) & (offers_export_pandas['spend_plus_stretch'] < 245),
    (offers_export_pandas['spend_plus_stretch'] >= 245) & (offers_export_pandas['spend_plus_stretch'] < 250),
    (offers_export_pandas['spend_plus_stretch'] >= 250) & (offers_export_pandas['spend_plus_stretch'] < 255),
    (offers_export_pandas['spend_plus_stretch'] >= 255) & (offers_export_pandas['spend_plus_stretch'] < 260),
    (offers_export_pandas['spend_plus_stretch'] >= 260) & (offers_export_pandas['spend_plus_stretch'] < 265),
    (offers_export_pandas['spend_plus_stretch'] >= 265) & (offers_export_pandas['spend_plus_stretch'] < 270),
]

offers_5 = [
    '£5 off when you spend £30 on M&S food in store',
    '£5 off when you spend £35 on M&S food in store',
    '£5 off when you spend £40 on M&S food in store',
    '£5 off when you spend £45 on M&S food in store',
    '£5 off when you spend £50 on M&S food in store',
    '£5.5 off when you spend £55 on M&S food in store',
    '£6 off when you spend £60 on M&S food in store',
    '£6.5 off when you spend £65 on M&S food in store',
    '£7 off when you spend £70 on M&S food in store',
    '£7.5 off when you spend £75 on M&S food in store',
    '£8 off when you spend £80 on M&S food in store',
    '£8.5 off when you spend £85 on M&S food in store',
    '£9 off when you spend £90 on M&S food in store',
    '£9.5 off when you spend £95 on M&S food in store',
    '£10 off when you spend £100 on M&S food in store',
    '£10.5 off when you spend £105 on M&S food in store',
    '£11 off when you spend £110 on M&S food in store',
    '£11.5 off when you spend £115 on M&S food in store',
    '£12 off when you spend £120 on M&S food in store',
    '£12.5 off when you spend £125 on M&S food in store',
    '£13 off when you spend £130 on M&S food in store',
    '£13.5 off when you spend £135 on M&S food in store',
    '£14 off when you spend £140 on M&S food in store',
    '£14.5 off when you spend £145 on M&S food in store',
    '£15 off when you spend £150 on M&S food in store',
    '£15.5 off when you spend £155 on M&S food in store',
    '£16 off when you spend £160 on M&S food in store',
    '£16.5 off when you spend £165 on M&S food in store',
    '£17 off when you spend £170 on M&S food in store',
    '£17.5 off when you spend £175 on M&S food in store',
    '£18 off when you spend £180 on M&S food in store',
    '£18.5 off when you spend £185 on M&S food in store',
    '£19 off when you spend £190 on M&S food in store',
    '£19.5 off when you spend £195 on M&S food in store',
    '£20 off when you spend £200 on M&S food in store',
    '£20.5 off when you spend £205 on M&S food in store',
    '£21 off when you spend £210 on M&S food in store',
    '£21.5 off when you spend £215 on M&S food in store',
    '£22 off when you spend £220 on M&S food in store',
    '£22.5 off when you spend £225 on M&S food in store',
    '£23 off when you spend £230 on M&S food in store',
    '£23.5 off when you spend £235 on M&S food in store',
    '£24 off when you spend £240 on M&S food in store',
    '£24.5 off when you spend £245 on M&S food in store',
    '£25 off when you spend £250 on M&S food in store',
    '£25.5 off when you spend £255 on M&S food in store',
    '£26 off when you spend £260 on M&S food in store',
    '£26.6 off when you spend £265 on M&S food in store',
    '£27 off when you spend £270 on M&S food in store'
]


# COMMAND ----------

import numpy as np
offers_export_pandas['desc_10'] = np.select(conditions_10, offers_10, default=np.nan)

# COMMAND ----------

offers_export_pandas['desc_5'] = np.select(conditions_5, offers_5, default=np.nan)

# COMMAND ----------

offers_export_pandas_10_20_stretch = offers_export_pandas[offers_export_pandas['test_type'].isin(['baseline_85_stretch_10_perc','baseline_85_stretch_20_perc'])]

# COMMAND ----------

number_of_offers_per_customer_10 = offers_export_pandas_10_20_stretch.groupby('cust_id')['desc_10'].nunique().reset_index(name='distinct_desc_10_count')

# COMMAND ----------

number_of_offers_per_customer_5 = offers_export_pandas_10_20_stretch.groupby('cust_id')['desc_5'].nunique().reset_index(name='distinct_desc_5_count')

# COMMAND ----------

number_of_offers_per_customer_20 = offers_export_pandas_10_20_stretch.groupby('cust_id')['desc'].nunique().reset_index(name='distinct_desc_count')

# COMMAND ----------

# Fill NaN values with a placeholder (like empty string or "Unknown")
number_of_offers_per_customer_5['cust_id'] = number_of_offers_per_customer_5['cust_id'].fillna('').astype(str)
offers_export_pandas['cust_id'] = offers_export_pandas['cust_id'].fillna('').astype(str)

# Strip any extra spaces again
number_of_offers_per_customer_5['cust_id'] = number_of_offers_per_customer_5['cust_id'].str.strip()
offers_export_pandas['cust_id'] = offers_export_pandas['cust_id'].str.strip()


# COMMAND ----------

# Now perform the join
offers_baseline_merged_5 = pd.merge(number_of_offers_per_customer_5,
    offers_export_pandas_10_20_stretch[offers_export_pandas_10_20_stretch['test_type'] == 'baseline_85_stretch_10_perc'][['cust_id','spend_plus_stretch']], on='cust_id', how = 'left'
)

# COMMAND ----------

offers_baseline_merged_10 = pd.merge(number_of_offers_per_customer_10,
    offers_export_pandas_10_20_stretch[offers_export_pandas_10_20_stretch['test_type'] == 'baseline_85_stretch_10_perc'][['cust_id','spend_plus_stretch']], on='cust_id', how = 'left'
)

# COMMAND ----------

offers_baseline_merged_20 = pd.merge(number_of_offers_per_customer_20,
    offers_export_pandas_10_20_stretch[offers_export_pandas_10_20_stretch['test_type'] == 'baseline_85_stretch_10_perc'][['cust_id','spend_plus_stretch']], on='cust_id', how = 'left'
)

# COMMAND ----------

import numpy as np
import pandas as pd

# Define bin edges starting from 0 with a width of 20 (common across all DataFrames)
bin_edges = np.arange(0, max(offers_baseline_merged_5['spend_plus_stretch'].max(),
                             offers_baseline_merged_10['spend_plus_stretch'].max(),
                             offers_baseline_merged_20['spend_plus_stretch'].max()) + 20, 20)

# Create labels based on the bin edges
labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in pd.cut(offers_baseline_merged_5['spend_plus_stretch'], bins=bin_edges, include_lowest=True).cat.categories]

# Apply the same bins and labels to each DataFrame
offers_baseline_merged_5['spend_bins'] = pd.cut(offers_baseline_merged_5['spend_plus_stretch'], bins=bin_edges, labels=labels, include_lowest=True)
offers_baseline_merged_10['spend_bins'] = pd.cut(offers_baseline_merged_10['spend_plus_stretch'], bins=bin_edges, labels=labels, include_lowest=True)
offers_baseline_merged_20['spend_bins'] = pd.cut(offers_baseline_merged_20['spend_plus_stretch'], bins=bin_edges, labels=labels, include_lowest=True)

# COMMAND ----------

offers_baseline_merged_grouped_20 = offers_baseline_merged_20[['spend_bins','distinct_desc_count']].groupby('spend_bins').mean('distinct_desc_count').reset_index()
offers_baseline_merged_grouped_10 = offers_baseline_merged_10[['spend_bins','distinct_desc_10_count']].groupby('spend_bins').mean('distinct_desc_10_count').reset_index()
offers_baseline_merged_grouped_5 = offers_baseline_merged_5[['spend_bins','distinct_desc_5_count']].groupby('spend_bins').mean('distinct_desc_5_count').reset_index()

# COMMAND ----------

# percentage of customers under £94 spend
all_customers = offers_export_pandas_10_20_stretch['cust_id'].unique()
customers_less_than_60 = offers_export_pandas_10_20_stretch[(offers_export_pandas_10_20_stretch['spend_plus_stretch'] < 60) & (offers_export_pandas_10_20_stretch['test_type'] =='baseline_85_stretch_10_perc')]['cust_id'].unique()
customers_60_to_200 = offers_export_pandas_10_20_stretch[(offers_export_pandas_10_20_stretch['spend_plus_stretch'] >= 60) & (offers_export_pandas_10_20_stretch['spend_plus_stretch'] < 200) & (offers_export_pandas_10_20_stretch['test_type'] =='baseline_85_stretch_10_perc')]['cust_id'].unique()

print(f"% of customers under £60 spend: {customers_less_than_60.size / all_customers.size * 100:.2f}%")
print(f"% of customers between £60-£200: {customers_60_to_200.size / all_customers.size * 100:.2f}%")
print(f"% of customers above £200 spend: {(1 - (customers_less_than_60.size / all_customers.size) - (customers_60_to_200.size / all_customers.size))*100:.2f}%")

# COMMAND ----------

conditions = [
    offers_export_pandas_10_20_stretch['cust_id'].isin(customers_less_than_60),
    offers_export_pandas_10_20_stretch['cust_id'].isin(customers_60_to_200)
]

# Define the corresponding outputs
choices = [
    offers_export_pandas_10_20_stretch['desc_5'],
    offers_export_pandas_10_20_stretch['desc_10']
]
offers_export_pandas_10_20_stretch['final_desc'] = np.select(
    conditions, 
    choices, 
    default=offers_export_pandas_10_20_stretch['desc']
)

# COMMAND ----------

number_of_offers_per_customer_mixed = offers_export_pandas_10_20_stretch.groupby('cust_id')['final_desc'].nunique().reset_index(name='distinct_desc_final_count')

# COMMAND ----------

offers_baseline_merged_mixed = pd.merge(number_of_offers_per_customer_mixed,
    offers_export_pandas_10_20_stretch[offers_export_pandas_10_20_stretch['test_type'] == 'baseline_85_stretch_10_perc'][['cust_id','spend_plus_stretch']], on='cust_id', how = 'left'
)

# COMMAND ----------

offers_baseline_merged_mixed['spend_bins'] = pd.cut(offers_baseline_merged_mixed['spend_plus_stretch'], bins=bin_edges, labels=labels, include_lowest=True)

# COMMAND ----------

offers_baseline_merged_mixed_grouped = offers_baseline_merged_mixed[['spend_bins','distinct_desc_final_count']].groupby('spend_bins').mean('distinct_desc_count').reset_index()

# COMMAND ----------

import matplotlib.pyplot as plt

# Set the figure size to make the plot clearer
plt.figure(figsize=(8, 6))

# Create the horizontal bar chart
plt.barh(offers_baseline_merged_mixed_grouped['spend_bins'], 
         offers_baseline_merged_mixed_grouped['distinct_desc_final_count'], 
         color='gray', alpha=0.7)

# Set the title and labels
plt.title('Distinct Offers per Customer (mixed offers)')
plt.xlabel('Average Number of Offers')

# Create a secondary y-axis
ax2 = plt.gca().twinx()

# Set the secondary y-axis limits to cover the range of the original y-axis
ax2.set_ylim(0, 280)

# Plot the horizontal lines on the secondary y-axis
ax2.axhline(y=60, color='b', linestyle='--', linewidth=2, label='£10 increment offers')
ax2.axhline(y=200, color='r', linestyle='--', linewidth=2, label='Original offers')
# ax2.axhline(y=0, color='g', linestyle='--', linewidth=2, label='£5 increment offers')

# Add legend for the lines on the secondary axis
ax2.legend(loc="upper right")

plt.tight_layout()

# Show the plot
plt.show()




import matplotlib.pyplot as plt

# Create a figure with 1 row and 2 columns for subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5)) # Adjust the figsize as needed

# First subplot

axes[0].barh(offers_baseline_merged_grouped_5['spend_bins'], offers_baseline_merged_grouped_5['distinct_desc_5_count'])
axes[0].set_title('Distinct Offers per Customer (5 increment offers)')
axes[0].set_xlabel('Average Number of Offers')
axes[0].set_ylabel('Spend plus stretch bins')
axes[0].tick_params(axis='y', labelsize=8)  # Adjust the y-axis font size


axes[1].barh(offers_baseline_merged_grouped_10['spend_bins'], offers_baseline_merged_grouped_10['distinct_desc_10_count'])
axes[1].set_title('Distinct Offers per Customer (10 increment offers)')
axes[1].set_xlabel('Average Number of Offers')
axes[1].set_ylabel('Spend plus stretch bins')
axes[1].tick_params(axis='y', labelsize=8)  # Adjust the y-axis font size

# Second subplot (you can plot another barh or same one with different data/parameters)
axes[2].barh(offers_baseline_merged_grouped_20['spend_bins'], offers_baseline_merged_grouped_20['distinct_desc_count'])
axes[2].set_title('Distinct Offers per Customer (current offers)')
axes[2].set_xlabel('Average Number of Offers')
axes[2].tick_params(axis='y', labelsize=8)  # Adjust the y-axis font size

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

# COMMAND ----------

offers_export_pandas_10_20_stretch['final_desc']

# COMMAND ----------

offers_export_pandas_10_20_stretch['amount_required_to_redeem'] = offers_export_pandas_10_20_stretch['final_desc'].str.extract(r'spend £(\d+)').astype(int)

# COMMAND ----------

offers_export_pandas_10_20_stretch['amount_required_to_redeem_5'] = offers_export_pandas_10_20_stretch['desc_5'].str.extract(r'spend £(\d+)').astype(int)

# COMMAND ----------

offers_export_pandas_10_20_stretch['amount_required_to_redeem_10'] = offers_export_pandas_10_20_stretch['desc_10'].str.extract(r'spend £(\d+)').astype(int)

# COMMAND ----------

offers_export_pandas_10_20_stretch['amount_required_to_redeem_20'] = offers_export_pandas_10_20_stretch['desc'].str.extract(r'spend £(\d+)').astype(int)

# COMMAND ----------

offers_export_pandas_10_20_stretch['additional_stretch_to_redeem'] = offers_export_pandas_10_20_stretch['amount_required_to_redeem'] - offers_export_pandas_10_20_stretch['spend_plus_stretch']
offers_export_pandas_10_20_stretch['additional_stretch_to_redeem_5'] = offers_export_pandas_10_20_stretch['amount_required_to_redeem_5'] - offers_export_pandas_10_20_stretch['spend_plus_stretch']
offers_export_pandas_10_20_stretch['additional_stretch_to_redeem_10'] = offers_export_pandas_10_20_stretch['amount_required_to_redeem_10'] - offers_export_pandas_10_20_stretch['spend_plus_stretch']
offers_export_pandas_10_20_stretch['additional_stretch_to_redeem_20'] = offers_export_pandas_10_20_stretch['amount_required_to_redeem_20'] - offers_export_pandas_10_20_stretch['spend_plus_stretch']

# COMMAND ----------

offers_export_pandas_10_20_stretch[['cust_id','spend_plus_stretch', 'final_desc', 'amount_required_to_redeem', 'additional_stretch_to_redeem']][offers_export_pandas_10_20_stretch['test_type'] == ''].head()

# COMMAND ----------

import pyplot as plt

# COMMAND ----------

number_of_offers_per_customer['distinct_desc_2_count'].mean()

# COMMAND ----------

offer_counts =  offers_export_pandas.pivot_table(index='test_type', columns='desc_2', aggfunc='size').reset_index()

# COMMAND ----------

offer_counts_5 =  offers_export_pandas.pivot_table(index='test_type', columns='desc_5', aggfunc='size').reset_index()

# COMMAND ----------

import pandas as pd

# Assuming df is your DataFrame

# Step 1: Convert all numerical values to percentages (excluding the first column)
df_percent_5 = offer_counts_5.set_index('test_type').apply(lambda x: 100 * x / x.sum(), axis=1)

# Step 2: Transpose the DataFrame
df_transposed_5 = df_percent_5.T

# To reset the index if necessary (optional)
df_transposed_5 = df_transposed_5.reset_index()

# Display the resulting DataFrame
df_transposed_5.display()

# COMMAND ----------

import pandas as pd

# Assuming df is your DataFrame

# Step 1: Define your specific order for the 'desc' column
desired_order_desc_5 = offers_5
# Step 2: Convert the 'desc' column to a categorical type with the desired order
df_transposed_5['desc_5'] = pd.Categorical(df_transposed_5['desc_5'], categories=desired_order_desc_5, ordered=True)

# Step 3: Sort the DataFrame based on the new 'desc' order
df_sorted_5 = df_transposed_5.sort_values('desc_5')

# Now you can perform further operations on the sorted DataFrame
df_sorted_5.display()

# COMMAND ----------

import pandas as pd

# Assuming df is your DataFrame

# Step 1: Convert all numerical values to percentages (excluding the first column)
df_percent = offer_counts.set_index('test_type').apply(lambda x: 100 * x / x.sum(), axis=1)

# Step 2: Transpose the DataFrame
df_transposed = df_percent.T

# To reset the index if necessary (optional)
df_transposed = df_transposed.reset_index()

# Display the resulting DataFrame
df_transposed

# COMMAND ----------

low_offers = ['16437', '16441', '16717','16445']
medium_offers = ['16442','16735', '16719', '16736','16718']
high_offers = ['16716', '16734', '16698','16699']

# COMMAND ----------

all_offers = low_offers + medium_offers + high_offers

# COMMAND ----------

plt.figure(figsize=(10, 6))

import pandas as pd
import matplotlib.pyplot as plt

# Group the data by offer_id and test_type, then count the occurrences
offer_counts_pivot =  offers_export_pandas.pivot_table(index='desc_2', columns='test_type', aggfunc='size', fill_value=0)

# Plot a stacked bar chart
offer_counts_pivot.plot(kind='barh', stacked=True)

# Add labels and title
# plt.xticks(ticks=range(len(offer_counts.index)), labels=['85th + 0 stretch', '85th + 10 stretch', '85th + 20 stretch', '85th + 30 stretch', 'headroom'], rotation=90)
plt.xlabel('Offer')
plt.ylabel('Number of offers')
plt.title('Frequency of offers per test cell')
plt.yticks(fontsize=8)
plt.legend(fontsize=8)
# Show the chart
plt.show()

# COMMAND ----------

offers_export_pandas

# COMMAND ----------

offer_counts =  offers_export_pandas.pivot_table(index='test_type', columns='desc_2', aggfunc='size').reset_index()


# COMMAND ----------

import pandas as pd

# Assuming df is your DataFrame

# Step 1: Convert all numerical values to percentages (excluding the first column)
df_percent = offer_counts.set_index('test_type').apply(lambda x: 100 * x / x.sum(), axis=1)

# Step 2: Transpose the DataFrame
df_transposed = df_percent.T

# To reset the index if necessary (optional)
df_transposed = df_transposed.reset_index()

# Display the resulting DataFrame
df_transposed



# COMMAND ----------

import pandas as pd

# Assuming df is your DataFrame

# Step 1: Define your specific order for the 'desc' column
desired_order_desc = offers_2
# Step 2: Convert the 'desc' column to a categorical type with the desired order
df_transposed['desc_2'] = pd.Categorical(df_transposed['desc_2'], categories=desired_order_desc, ordered=True)

# Step 3: Sort the DataFrame based on the new 'desc' order
df_sorted = df_transposed.sort_values('desc_2')

# Now you can perform further operations on the sorted DataFrame
df_sorted

# COMMAND ----------

df_transposed.display()

# COMMAND ----------

desc_order = [
  '£5 off when you spend £30 on M&S food in store',
 '£5 off when you spend £50 on M&S food in store',
 '£7 off when you spend £70 on M&S food in store',
 '£9 off when you spend £90 on M&S food in store',
 '£11 off when you spend £110 on M&S food in store',
 '£13 off when you spend £130 on M&S food in store',
 '£15 off when you spend £150 on M&S food in store',
 '£17 off when you spend £170 on M&S food in store',
 '£19 off when you spend £190 on M&S food in store',
 '£21 off when you spend £210 on M&S food in store',
 '£23 off when you spend £230 on M&S food in store',
 '£25 off when you spend £250 on M&S food in store',
 '£27 off when you spend £270 on M&S food in store']

# COMMAND ----------

df_sorted.display()

# COMMAND ----------



# COMMAND ----------

df_sorted.display()

# COMMAND ----------

plt.figure(figsize=(10, 6))

import pandas as pd
import matplotlib.pyplot as plt

# Group the data by offer_id and test_type, then count the occurrences
offer_counts =  offers_export_pandas[offers_export_pandas['offer_id'].isin(low_offers)].pivot_table(index='test_type', columns='desc', aggfunc='size', fill_value=0)

# Plot a stacked bar chart
offer_counts.plot(kind='bar', stacked=True)

# Add labels and title
plt.xticks(ticks=range(len(offer_counts.index)), labels=['0% stretch', '10% stretch', '20% stretch', '30% stretch', 'headroom'], rotation=0, fontsize=8)
plt.xlabel('Test cell')
plt.ylabel('Number of offers allocated')
plt.title('Allocation of low offers for each test cell')
plt.grid(False)
plt.legend(fontsize=8)
# Show the chart
plt.show()





# COMMAND ----------

plt.figure(figsize=(10, 6))

import pandas as pd
import matplotlib.pyplot as plt

# Group the data by offer_id and test_type, then count the occurrences
offer_counts =  offers_export_pandas[offers_export_pandas['offer_id'].isin(medium_offers)].pivot_table(index='test_type', columns='desc', aggfunc='size', fill_value=0)

# Plot a stacked bar chart
offer_counts.plot(kind='bar', stacked=True)

# Add labels and title
plt.xticks(ticks=range(len(offer_counts.index)), labels=['0% stretch', '10% stretch', '20% stretch', '30% stretch', 'headroom'], rotation=0, fontsize=8)
plt.xlabel('Test cell')
plt.ylabel('Number of offers allocated')
plt.title('Allocation of medium offers for each test cell')
plt.legend(fontsize=8)
plt.grid(False)
# Show the chart
plt.show()


# COMMAND ----------

plt.figure(figsize=(10, 6))

import pandas as pd
import matplotlib.pyplot as plt

# Group the data by offer_id and test_type, then count the occurrences
offer_counts =  offers_export_pandas[offers_export_pandas['offer_id'].isin(high_offers)].pivot_table(index='test_type', columns='desc', aggfunc='size', fill_value=0)

# Plot a stacked bar chart
offer_counts.plot(kind='bar', stacked=True)

# Add labels and title
plt.xticks(ticks=range(len(offer_counts.index)), labels=['0% stretch', '10% stretch', '20% stretch', '30% stretch', 'headroom'], rotation=0, fontsize=8)
plt.xlabel('Test cell')
plt.ylabel('Number of offers allocated')
plt.title('Allocation of high offers for each test cell')
plt.legend(fontsize=8)
plt.grid(False)
# Show the chart
plt.show()


# COMMAND ----------

# Offer allocation table
fixed_stretch_tbl_name = persist_utils.get_table_name(
    factory_database=config.factory_database,
    lab_database=config.lab_database,
    table_prefix=config_al.fixed_stretch_tbl.prefix,
    sensitivity=config.sensitivity,
)
logger.info(f"""fixed_stretch_tbl_name: {fixed_stretch_tbl_name}""")

fixed_stretch_tbl = persist_utils.read_table(
    table_name=fixed_stretch_tbl_name)

# COMMAND ----------

trans_before_june_accu = spark.sql(
  """
  select
  *
  from 
  loyalty_azlab_prod.legacy_all_transaction_line
  where EVENT_DATE >= '2023-06-01'
  and EVENT_DATE <= '2024-05-31'
  and sparks_reg_date <= '2023-06-01'
  and trans_line_type = 'S'
  and division_id = 'FD'
  and sparks_account_id is not null
    """)

# COMMAND ----------

trans_before_june_accu.columns

# COMMAND ----------

trans_before_june_accu.display()
