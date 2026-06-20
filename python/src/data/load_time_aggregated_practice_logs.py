from typing import Optional

import numpy as np
import polars as pl

from polars import DataFrame, Series

from data.load_practice_logs_for_author import load_practice_logs_for_author
from functions.calc_message_embeddings import add_message_embeddings
from models.EmbeddingModelABC import EmbeddingModelABC
from models.SentenceTransformerModel import SentenceTransformerModel


def main():
    df = load_time_aggregated_practice_logs(
        time_aggregate_every="1w",
        author="Linda ”Polly Ester” Ö",
        model=SentenceTransformerModel("all-MiniLM-L6-v2"),
    )
    print(df)


def load_time_aggregated_practice_logs(
        author: str,
        model: EmbeddingModelABC,
        time_aggregate_every: str,
        time_aggregate_period: Optional[str] = None) -> DataFrame:
    df = load_practice_logs_for_author(author=author)
    df = add_message_embeddings(df=df, model=model)
    return aggregate_messages_by_time(df=df, every=time_aggregate_every, period=time_aggregate_period)


def aggregate_messages_by_time(df: DataFrame, every: str, period: Optional[str] = None) -> DataFrame:
    embedding_dim = df["embedding"].dtype.shape[0]
    return df.select(
        ["date", "msg", "embedding"]
    ).sort(
        "date"
    ).group_by_dynamic(
        "date", every=every, period=period
    ).agg(
        pl.col("msg").str.concat(" "),
        pl.col("embedding").map_batches(
            _mean_pool,
            agg_list=True, return_dtype=pl.Array(pl.Float32, embedding_dim)
        ).alias("embedding"),
    )


def _mean_pool(s: Series):
    return np.vstack([np.array(x).mean(axis=0) for x in s])


if __name__ == "__main__":
    main()
