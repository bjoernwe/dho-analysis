import numpy as np
import polars as pl

from polars import DataFrame, Series

from dho_analysis.load_practice_logs_for_author import load_practice_logs_for_author


def main():
    df = load_time_aggregated_practice_logs(time_aggregate="1w", author="Linda ”Polly Ester” Ö")
    print(df)


def load_time_aggregated_practice_logs(time_aggregate: str, author: str) -> DataFrame:
    df = load_practice_logs_for_author(author=author)
    return aggregate_messages_by_time(df=df, time_aggregate=time_aggregate)


def aggregate_messages_by_time(df: DataFrame, time_aggregate: str) -> DataFrame:
    embedding_dim = df["embedding"].dtype.shape[0]
    return df.select(
        ["date", "msg", "embedding"]
    ).sort(
        "date"
    ).group_by_dynamic(
        "date", every=time_aggregate
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
