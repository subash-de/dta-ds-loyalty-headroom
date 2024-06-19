# Databricks notebook source
# TODO: WIP Beam write class, use this later
class BeamWriter:
  def __init__(self, default_config):
    self.default_config = default_config

  def write_beam_table(
    data,
    table_info,
    add_timestamp=True,
  ):
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
    return data

