import polars as pl

from polars import DataFrame


def read_dho_messages() -> DataFrame:
    return pl.read_ndjson(
        "../../data/messages.jsonl"
    ).with_columns(
        pl.col("date").str.strptime(pl.Datetime)
    )
