import numpy as np

from typing import Tuple

from polars import DataFrame, Series
from sentence_transformers import SentenceTransformer

from dho_analysis.load_time_aggregated_practice_logs import load_time_aggregated_practice_logs
from dho_analysis.utils import memory, CACHE_DIR


def main():
    df = load_time_aggregated_practice_logs(time_aggregate="1w", author="Linda ”Polly Ester” Ö")
    df = add_message_embeddings(df)
    print(df)


def add_message_embeddings(df: DataFrame, model: str = "all-MiniLM-L6-v2", normalize: bool = False) -> DataFrame:
    """
    Models:
    - all-MiniLM-L6-v2
    - all-mpnet-base-v2
    - msmarco-MiniLM-L6-cos-v5
    - AnnaWegmann/Style-Embedding
    """
    sentences = tuple(df.get_column("msg").to_list())
    embeddings = _calc_embeddings(sentences, model=model, normalize=normalize)
    return df.with_columns(
        Series(embeddings).alias("embedding")
    )


# Changes to this function will invalidate its cache!
@memory.cache
def _calc_embeddings(sentences: Tuple[str], model: str, normalize: bool = False) -> np.ndarray:
    model = SentenceTransformer(model, cache_folder=CACHE_DIR)
    embeddings = model.encode(list(sentences), show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=normalize)
    return embeddings


if __name__ == "__main__":
    main()
