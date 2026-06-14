import numpy as np
from polars import DataFrame, Series

from config import SEED
from data.load_practice_logs_for_author import load_practice_logs_for_author
from models.EmbeddingModelABC import EmbeddingModelABC
from models.SentenceTransformerModel import SentenceTransformerModel


generator = np.random.default_rng(SEED)


def main():
    print_example_embeddings()


def print_example_embeddings():
    model = SentenceTransformerModel("all-MiniLM-L6-v2")
    df = load_practice_logs_for_author(author="Linda ”Polly Ester” Ö")
    df = add_message_embeddings(df, model=model)
    print(df)


def add_message_embeddings(df: DataFrame, model: EmbeddingModelABC) -> DataFrame:
    embeddings = model.encode(df.get_column("msg"))
    return df.with_columns(
        Series(embeddings).alias("embedding")
    )


def add_mock_message_embeddings(df: DataFrame, dims: int = 10) -> DataFrame:
    embeddings = Series(generator.random((len(df), dims)))
    return df.with_columns(
        Series(embeddings).alias("embedding")
    )



if __name__ == "__main__":
    main()
