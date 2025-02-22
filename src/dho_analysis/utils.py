from pathlib import Path

import polars as pl
from joblib import Memory

from polars import DataFrame


project_path: Path = Path(__file__).parent.parent.parent
cache_dir: str = str(project_path.joinpath(".cache"))

memory = Memory(location=cache_dir)


def read_dho_messages() -> DataFrame:
    return pl.read_ndjson(
        "../../data/messages.jsonl"
    ).with_columns(
        pl.col("date").str.strptime(pl.Datetime)
    )
