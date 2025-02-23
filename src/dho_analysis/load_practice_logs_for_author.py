import polars as pl

from polars import DataFrame

from dho_analysis.utils import read_dho_messages


def main():
    df = load_practice_logs_for_author(author="Linda ”Polly Ester” Ö")
    print(df)


def load_practice_logs_for_author(author: str) -> DataFrame:
    return read_dho_messages().filter(
        pl.col("category").eq("PracticeLogs"),
        pl.col("author").eq(author),
    )


if __name__ == "__main__":
    main()
