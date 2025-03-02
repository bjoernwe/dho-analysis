import nltk
import polars as pl

from typing import List

from polars import DataFrame

from data.load_practice_logs_for_author import load_practice_logs_for_author


nltk.download("punkt_tab")


def main():
    print_example_sentences()


def print_example_sentences():
    df = load_practice_logs_for_author(author="Linda ”Polly Ester” Ö")
    df = explode_msg_to_sentences(df=df)
    print(df)


def explode_msg_to_sentences(df: DataFrame) -> DataFrame:
    return (
        _split_msg_to_sentence_list(df=df)
        .explode("msg")
        .filter(pl.col("msg").is_not_null())
    )


def _split_msg_to_sentence_list(df: DataFrame) -> DataFrame:
    return df.with_columns(
        pl.col("msg").map_elements(_split_sentences, return_dtype=pl.List(pl.String))
    )


def _split_sentences(paragraph: str) -> List[str]:
    return nltk.sent_tokenize(paragraph)


if __name__ == "__main__":
    main()
