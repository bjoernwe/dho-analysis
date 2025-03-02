import functools

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel, PreTrainedTokenizer


def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@functools.lru_cache
def get_tokenizer(model_name: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


@functools.lru_cache
def get_model_for_sequence_classification(model_name: str) -> PreTrainedModel:
    return (
        AutoModelForSequenceClassification
        .from_pretrained(model_name)
        .to(get_device())
    )
