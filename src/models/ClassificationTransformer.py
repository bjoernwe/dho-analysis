import numpy as np
import torch

from typing import Tuple

from polars import Series
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import memory
from models.EmbeddingModelABC import EmbeddingModelABC


class ClassificationTransformer(EmbeddingModelABC):

    def __init__(self, model: str, batch_size: int = 1000):
        self._model_name: str = model
        self._batch_size: int = batch_size

    def encode(self, msgs: Series) -> Series:
        results = []
        for i in range(0, len(msgs), self._batch_size):
            batch: Tuple = tuple(msgs[i:i+self._batch_size].to_list())
            embeddings = _calc_embeddings(msgs=batch, model_name=self._model_name)
            results.append(embeddings)
        return Series("embedding", np.vstack(results))


# Changes to this function will invalidate its cache!
@memory.cache
def _calc_embeddings(msgs: Tuple[str], model_name: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    inputs = tokenizer(msgs, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs.detach().numpy()


def main():
    run_example()


def run_example():
    msgs = Series(["Exploding lights!", "Annoying dissonances!"])
    model = ClassificationTransformer(model="j-hartmann/emotion-english-distilroberta-base")
    output = model.encode(msgs=msgs)
    print(output)


if __name__ == "__main__":
    main()
