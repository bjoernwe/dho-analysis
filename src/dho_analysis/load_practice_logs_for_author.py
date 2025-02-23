from typing import Set

import polars as pl

from polars import DataFrame

from dho_analysis.utils import read_dho_messages


def main():
    df = load_practice_logs_for_author(author="Linda ”Polly Ester” Ö")
    print(df)


def load_practice_logs_for_author(author: str) -> DataFrame:
    dho_messages = read_dho_messages()
    practice_threads_by_author = _get_practice_thread_ids_by_author(df=dho_messages, author=author)
    return dho_messages.filter(
        pl.col("category").eq("PracticeLogs"),
        pl.col("author").eq(author),
        pl.col("thread_id").is_in(practice_threads_by_author),
    )


def _get_practice_thread_ids_by_author(df: DataFrame, author: str) -> Set[int]:
    return set(
        df.filter(
            pl.col("category").eq("PracticeLogs"),
            pl.col("author").eq(author),
            pl.col("is_first_in_thread"),
        ).get_column("thread_id")
    )


if __name__ == "__main__":
    main()
