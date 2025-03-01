import numpy as np

from typing import Tuple

from polars import Series
from sentence_transformers import SentenceTransformer

from models.EmbeddingModelABC import EmbeddingModelABC
from config import memory


class SentenceTransformerModel(EmbeddingModelABC):

    def __init__(self, model: str):
        self._model_name: str = model

    def encode(self, msgs: Series) -> Series:
        msg_tuple = tuple(msgs.to_list())
        embeddings = _calc_embeddings(msg_tuple, model=self._model_name)
        return Series(embeddings)


# Changes to this function will invalidate its cache!
@memory.cache
def _calc_embeddings(sentences: Tuple[str], model: str) -> np.ndarray:
    model = SentenceTransformer(model)
    embeddings = model.encode(list(sentences), show_progress_bar=True, convert_to_numpy=True)
    return embeddings
