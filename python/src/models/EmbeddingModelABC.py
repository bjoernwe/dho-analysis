from abc import ABC, abstractmethod

from polars import Series


class EmbeddingModelABC(ABC):

    @abstractmethod
    def encode(self, msgs: Series) -> Series:
        raise NotImplementedError()
