import polars as pl

from dho_analysis.utils import read_dho_messages


def main():
    df = calc_num_practice_logs_per_author()
    print(df)


def calc_num_practice_logs_per_author():
    return read_dho_messages().filter(
        pl.col("category").eq("PracticeLogs")
    ).group_by(
        "author"
    ).agg(
        pl.len().alias("num_posts"),
        pl.col("date").sort().first().dt.date().alias("first_date"),
        pl.col("date").sort().last().dt.date().alias("last_date"),
    ).sort(
        pl.col("num_posts"), descending=True
    )


if __name__ == "__main__":
    main()
