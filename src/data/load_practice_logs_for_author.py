from typing import Dict

import polars as pl

from polars import DataFrame

from config import read_dho_messages


def main():
    df = load_practice_logs_for_author(author="Linda ”Polly Ester” Ö")
    print(df)


def load_practice_logs_for_author(author: str) -> DataFrame:
    df = read_dho_messages()
    df = filter_for_thread_author_only(df=df)
    return df.filter(
        pl.col("category").eq("PracticeLogs"),
        pl.col("author").eq(author),
    )


def filter_for_thread_author_only(df: DataFrame) -> DataFrame:
    df = add_thread_author_column(df=df)
    return df.filter(
        pl.col("author").eq(pl.col("thread_author"))
    ).drop("thread_author")


def add_thread_author_column(df: DataFrame) -> DataFrame:
    thread_author_map = _get_thread_author_map(df=df)
    return df.with_columns(
        pl.col("thread_id").replace_strict(thread_author_map).alias("thread_author")
    )


def _get_thread_author_map(df: DataFrame) -> Dict[int, str]:
    df_thread_author = df.filter(
        pl.col("is_first_in_thread").eq(True)
    ).select(["thread_id", "author"])
    return dict(zip(
        df_thread_author["thread_id"].to_list(),
        df_thread_author["author"].to_list()
    ))


if __name__ == "__main__":
    main()
