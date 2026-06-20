from pathlib import Path

import polars as pl
from joblib import Memory

from polars import DataFrame


SEED = 0

PROJECT_PATH: Path = Path(__file__).parent.parent

CACHE_DIR: str = str(PROJECT_PATH.joinpath("cache"))
#CACHE_DIR: str = str(PROJECT_PATH.parent.joinpath("drive/MyDrive/Datasets/joblib-cache"))

DATA_PATH: Path = PROJECT_PATH.joinpath("data")
#DATA_PATH: Path = PROJECT_PATH.parent.joinpath(Path("drive/MyDrive/Datasets/dho-scrapy"))


memory = Memory(location=CACHE_DIR)


def read_dho_messages() -> DataFrame:
    return pl.read_ndjson(
        str(DATA_PATH.joinpath("messages.jsonl"))
    ).with_columns(
        pl.col("date").str.strptime(pl.Datetime)
    )
