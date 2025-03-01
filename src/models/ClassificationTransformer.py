import numpy as np

from typing import Tuple

from polars import Series
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from config import memory
from models.EmbeddingModelABC import EmbeddingModelABC


class ClassificationTransformer(EmbeddingModelABC):

    def __init__(self, model: str):
        self._model_name: str = model

    def encode(self, msgs: Series) -> Series:
        msgs_tuple = tuple(msgs.to_list())
        embeddings = _calc_embeddings(msgs=msgs_tuple, model_name=self._model_name)
        return Series(embeddings)


# Changes to this function will invalidate its cache!
@memory.cache
def _calc_embeddings(msgs: Tuple[str], model_name: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, )
    inputs = tokenizer(msgs, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.detach().numpy()


def main():
    run_example()


def run_example():
    msgs = Series(["Exploding lights!", "Annoying dissonances!"])
    model = ClassificationTransformer(model="j-hartmann/emotion-english-distilroberta-base")
    output = model.encode(msgs=msgs)
    print(output)


if __name__ == "__main__":
    main()
