import numpy as np

from typing import Tuple

from polars import Series
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from models.EmbeddingModelABC import EmbeddingModelABC
from config import memory


class SentenceTransformerModel(EmbeddingModelABC):
    """
    Models:
    - all-MiniLM-L6-v2
    - all-mpnet-base-v2
    - multi-qa-mpnet-base-dot-v1
    - msmarco-MiniLM-L6-cos-v5
    - AnnaWegmann/Style-Embedding
    """

    def __init__(self, model: str, batch_size: int = 1000):
        self._model_name: str = model
        self._batch_size = batch_size

    def encode(self, msgs: Series) -> Series:
        result = []
        for i in tqdm(range(0, len(msgs), self._batch_size)):
            batch: Tuple = tuple(msgs[i:i+self._batch_size].to_list())
            embeddings = _calc_embeddings(batch, model=self._model_name)
            result.append(embeddings)
        result = [item for sublist in result for item in sublist]  # flatten list
        return Series(result)


# Changes to this function will invalidate its cache!
@memory.cache
def _calc_embeddings(sentences: Tuple[str, ...], model: str) -> np.ndarray:
    model = SentenceTransformer(model)
    embeddings = model.encode(list(sentences), show_progress_bar=False, convert_to_numpy=True)
    return embeddings


def main():
    print_example_embedding()


def print_example_embedding():
    msgs = Series(["Exploding lights!", "Annoying dissonances!"])
    model = SentenceTransformerModel(model="all-MiniLM-L6-v2")
    output = model.encode(msgs=msgs)
    print(output)


if __name__ == "__main__":
    main()
