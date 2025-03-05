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
        series = [self._calc_scores_for_series(msgs=msgs, label=label) for label in self._labels]
        return Series("embedding", np.array(series))

    def _calc_scores_for_series(self, msgs: Series, label: str) -> Series:
        results = []
        for i in tqdm(range(0, len(msgs), self._batch_size)):
            batch: Tuple = tuple(msgs[i:i+self._batch_size].to_list())
            embeddings = _calc_scores(msgs=batch, model_name=self._model_name, label=label)
            results.append(embeddings)
        results = [item for sublist in results for item in sublist]  # flatten list
        return Series(f"score ({label})", results)


# Changes to this function will invalidate its cache!
@memory.cache
def _calc_scores(msgs: Tuple[str, ...], model_name: str, label: str) -> List[float]:
    return [_calc_score(msg=msg, model_name=model_name, label=label) for msg in msgs]


def _calc_score(msg: str, model_name: str, label: str) -> float:
    classifier = get_pipeline(pipeline_type="zero-shot-classification", model_name=model_name)
    outputs = classifier([msg], [label], multi_label=True)
    return outputs[0]["scores"][0]


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
