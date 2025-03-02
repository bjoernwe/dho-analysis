from typing import Tuple, List

import numpy as np

from polars import Series
from tqdm import tqdm

from config import memory
from models.EmbeddingModelABC import EmbeddingModelABC
from models.TransformerModelUtils import get_pipeline


class ZeroShotEmbeddingTransformer(EmbeddingModelABC):

    def __init__(self, model: str, labels: List[str], batch_size: int = 100):
        self._model_name: str = model
        self._labels: List[str] = labels
        self._batch_size: int = batch_size

    def encode(self, msgs: Series) -> Series:
        results = []
        for i in tqdm(range(0, len(msgs), self._batch_size)):
            batch: Tuple = tuple(msgs[i:i+self._batch_size].to_list())
            embeddings = _calc_embeddings(msgs=batch, model_name=self._model_name, labels=tuple(self._labels))
            results.append(embeddings)
        return Series("embedding", np.vstack(results))


# Changes to this function will invalidate its cache!
@memory.cache
def _calc_embeddings(msgs: Tuple[str, ...], model_name: str, labels: Tuple[str, ...]) -> np.ndarray:

    classifier = get_pipeline(pipeline_type="zero-shot-classification", model_name=model_name)

    outputs = classifier(list(msgs), list(labels), multi_label=True)
    scores = np.array([_output_to_sorted_scores(out) for out in outputs])

    # restore original label order
    idc = np.argsort(labels)
    return scores[:,idc]


def _output_to_sorted_scores(output: dict) -> np.array:
    label_indices = np.argsort(output["labels"])
    return np.array(output["scores"])[label_indices]


def main():
    print_example_embedding()


def print_example_embedding():
    model = ZeroShotEmbeddingTransformer(
        model="facebook/bart-large-mnli",
        labels=["positive", "negative", "furry"]
    )
    output = model.encode(Series(["This is cool!", "This is not cool!", "This is a cat"]))
    print(output)


if __name__ == "__main__":
    main()
