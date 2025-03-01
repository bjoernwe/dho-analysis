from polars import DataFrame, Series

from dho_analysis.load_practice_logs_for_author import load_practice_logs_for_author
from dho_analysis.models.SentenceTransformerModel import SentenceTransformerModel


def main():
    model = SentenceTransformerModel("all-MiniLM-L6-v2")
    df = load_practice_logs_for_author(author="Linda ”Polly Ester” Ö")
    df = add_message_embeddings(df, model=model)
    print(df)


def add_message_embeddings(df: DataFrame, model: SentenceTransformerModel) -> DataFrame:
    """
    Models:
    - all-MiniLM-L6-v2
    - all-mpnet-base-v2
    - msmarco-MiniLM-L6-cos-v5
    - AnnaWegmann/Style-Embedding
    """
    embeddings = model.encode(df.get_column("msg"))
    return df.with_columns(
        Series(embeddings).alias("embedding")
    )


if __name__ == "__main__":
    main()
