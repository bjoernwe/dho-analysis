import polars as pl

from polars import DataFrame

from dho_analysis.utils import read_dho_messages


def main():
    df = load_time_aggregated_practice_logs()
    print(df)


def load_time_aggregated_practice_logs(time_aggregate: str = "1d") -> DataFrame:
    return read_dho_messages().filter(
        pl.col("category").eq("PracticeLogs"),
        pl.col("author").eq("Linda ”Polly Ester” Ö")
    ).sort(
        "date"
    ).group_by_dynamic("date", every=time_aggregate).agg(
        pl.col("msg").str.concat(" ")
    )


if __name__ == "__main__":
    main()
