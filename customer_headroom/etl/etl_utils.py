from datetime import datetime
from pyspark.sql import functions as F
from dtaml.utils.table import factory_table
from customer_headroom.utils import persist_utils




def write_beam_table(
  data,
  config,
  table,
  add_timestamp=True,
):
  table_info = config.tables[table]
  table_name = factory_table(
    table_prefix=table_info.prefix,
    sensitivity=table_info.get("sensitivity", config.sensitivity) # Override default sensitivity
  )
  if add_timestamp:
    data = data.withColumn("load_timestamp", F.current_timestamp())
  data.save_beam_table(
    table_name, 
    format='delta', 
    mode='append', 
    partitionBy=table_info.get("partitionByList", None) # Partition by list of columns only if this property is set
  )
  return table_name

def find_all_segments(data, partitionByList):
    segs = (
        data.select(partitionByList)
        .distinct()
        .rdd.map(lambda x: {k: v for (k, v) in zip(partitionByList, x)})
        .collect()
    )
    return segs


def get_date(date):
    if str(date).lower() == "today":
        date = datetime.now().strftime("%Y%m%d")
    return int(date)


def get_campaign(campaign, etl_date):
    if (campaign == "{campaign}") or (campaign == ""):
        campaign = get_date(etl_date)
    return int(campaign)


def get_count(seg, config):
    partitionByList = seg.keys()
    seg_ext = [f"({k}='{seg[k]}')" for k in partitionByList]
    etl_data_tbl_name = factory_table(
        table_prefix=config.tables.etl_data_tbl.prefix,
        sensitivity=config.sensitivity
    )

    seg_etl_data_tbl = persist_utils.read_table(
        table_name=etl_data_tbl_name, where=" and ".join(seg_ext)
    )

    cnt = seg_etl_data_tbl.count()

    return (seg, cnt)