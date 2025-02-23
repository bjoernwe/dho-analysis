import nltk
import polars as pl

from typing import List

from polars import DataFrame

from dho_analysis.load_practice_logs_for_author import load_practice_logs_for_author


nltk.download("punkt_tab")


def main():
    df = load_practice_logs_for_author(author="Linda ”Polly Ester” Ö")
    df = explode_msg_to_sentences(df=df)
    print(df)


def explode_msg_to_sentences(df: DataFrame) -> DataFrame:
    return (split_msg_to_sentences(df=df)
            .explode("sentences")
            .rename({"sentences": "sentence"})
            )


def split_msg_to_sentences(df: DataFrame) -> DataFrame:
    return df.with_columns(
        pl.col("msg").map_elements(_split_sentences, return_dtype=pl.List(pl.String))
    ).rename(
        {"msg": "sentences"}
    )


def _split_sentences(paragraph: str) -> List[str]:
    return nltk.sent_tokenize(paragraph)


if __name__ == "__main__":
    main()
