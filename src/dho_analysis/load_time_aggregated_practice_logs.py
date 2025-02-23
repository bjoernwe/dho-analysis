import polars as pl

from polars import DataFrame

from dho_analysis.load_practice_logs_for_author import load_practice_logs_for_author


def main():
    df = load_time_aggregated_practice_logs(time_aggregate="1w", author="Linda ”Polly Ester” Ö")
    print(df)


def load_time_aggregated_practice_logs(time_aggregate: str, author: str) -> DataFrame:
    return load_practice_logs_for_author(
        author=author
    ).sort(
        "date"
    ).group_by_dynamic(
        "date", every=time_aggregate
    ).agg(
        pl.col("msg").str.concat(" ")
    )


if __name__ == "__main__":
    main()
