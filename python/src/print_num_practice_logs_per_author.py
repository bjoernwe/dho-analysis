import polars as pl

from data.load_practice_logs_for_author import filter_for_thread_author_only
from config import read_dho_messages


def main():
    df = load_num_practice_logs_per_author()
    print(df.head(10))


def load_num_practice_logs_per_author():
    df = read_dho_messages()
    df = filter_for_thread_author_only(df=df)
    return df.filter(
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
