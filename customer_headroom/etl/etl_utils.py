from datetime import datetime


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


def get_count(seg, config, database):
    partitionByList = seg.keys()
    seg_ext = [f"({k}='{seg[k]}')" for k in partitionByList]

    etl_data_tbl_name = persist_utils.get_table_name(
        factory_database=config.etl_data_tbl.factory_database,
        lab_database=database,
        table_prefix=config.etl_data_tbl.prefix,
        sensitivity=config.etl_data_tbl.sensitivity,
    )

    seg_etl_data_tbl = persist_utils.read_table(
        table_name=etl_data_tbl_name, where=" and ".join(seg_ext)
    )

    cnt = seg_etl_data_tbl.count()

    return (seg, cnt)