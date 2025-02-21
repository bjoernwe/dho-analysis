from polars import DataFrame, Series
from sentence_transformers import SentenceTransformer

from dho_analysis.time_aggregates import aggregate_time
from dho_analysis.utils import project_path


def main():
    df = aggregate_time()
    df = calc_message_embeddings(df)
    print(df)


def calc_message_embeddings(df: DataFrame, model: str = "all-MiniLM-L6-v2") -> DataFrame:
    model = SentenceTransformer(model, cache_folder=project_path.joinpath(".cache").__str__())
    embeddings = model.encode(df.get_column("msg").to_list(), show_progress_bar=True, convert_to_numpy=True)
    return df.with_columns(
        Series(embeddings).alias("embeddings")
    )


if __name__ == "__main__":
    main()
