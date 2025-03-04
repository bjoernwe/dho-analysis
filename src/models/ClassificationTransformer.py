import numpy as np
import torch

from typing import Tuple

from polars import Series
from tqdm import tqdm

from config import memory
from models.EmbeddingModelABC import EmbeddingModelABC
from models.TransformerModelUtils import get_tokenizer, get_model_for_sequence_classification


class ClassificationTransformer(EmbeddingModelABC):
    """
    Models:
    - j-hartmann/emotion-english-distilroberta-base
    - SamLowe/roberta-base-go_emotions
    - arpanghoshal/EmoRoBERTa
    """

    def __init__(self, model: str, batch_size: int = 100):
        self._model_name: str = model
        self._batch_size: int = batch_size

    def encode(self, msgs: Series) -> Series:
        results = []
        for i in tqdm(range(0, len(msgs), self._batch_size)):
            batch: Tuple = tuple(msgs[i:i+self._batch_size].to_list())
            embeddings = _calc_embeddings(msgs=batch, model_name=self._model_name)
            results.append(embeddings)
        return Series("embedding", np.vstack(results))


# Changes to this function will invalidate its cache!
@memory.cache
def _calc_embeddings(msgs: Tuple[str], model_name: str) -> np.ndarray:
    tokenizer = get_tokenizer(model_name=model_name)
    model = get_model_for_sequence_classification(model_name=model_name)
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    inputs = tokenizer(list(msgs), padding=True, truncation=True, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs.detach().cpu().numpy()


def main():
    run_example()


def run_example():
    msgs = Series(["Exploding lights!", "Annoying dissonances!"])
    model = ClassificationTransformer(model="j-hartmann/emotion-english-distilroberta-base")
    output = model.encode(msgs=msgs)
    print(output)


if __name__ == "__main__":
    main()
