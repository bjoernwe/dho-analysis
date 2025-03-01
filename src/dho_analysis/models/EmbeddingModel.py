from abc import ABC, abstractmethod

from polars import Series


class EmbeddingModel(ABC):

    @abstractmethod
    def encode(self, msgs: Series) -> Series:
        raise NotImplementedError()
