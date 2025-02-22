import numpy as np

from typing import Tuple

from polars import DataFrame, Series
from sentence_transformers import SentenceTransformer

from dho_analysis.load_time_aggregated_practice_logs import load_time_aggregated_practice_logs
from dho_analysis.utils import memory, cache_dir


def main():
    df = load_time_aggregated_practice_logs()
    df = add_message_embeddings(df)
    print(df)


# models
# - all-MiniLM-L6-v2
# - AnnaWegmann/Style-Embedding
def add_message_embeddings(df: DataFrame, model: str = "all-MiniLM-L6-v2") -> DataFrame:
    sentences = tuple(df.get_column("msg").to_list())
    embeddings = _calc_embeddings(sentences, model=model, normalize=True)
    return df.with_columns(
        Series(embeddings).alias("embedding")
    )


# Changes to this function will invalidate its cache!
@memory.cache
def _calc_embeddings(sentences: Tuple[str], model: str, normalize: bool = False) -> np.ndarray:
    model = SentenceTransformer(model, cache_folder=cache_dir)
    embeddings = model.encode(list(sentences), show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=normalize)
    return embeddings


if __name__ == "__main__":
    main()
