import numpy as np
import numpy.typing as npt
from hashlib import sha1
from typing import TypeVar, Type

HashArray = TypeVar("HashArray", bound="HashableNdarray")


class HashableNdarray(np.ndarray):
    """
    Class that makes a numpy array hashable, so that it can be used as an element
    in a set (or key in a dictionary)
    """

    @classmethod
    def create(cls: Type[HashArray], array: npt.NDArray[np.int_]) -> HashArray:
        return HashableNdarray(
            shape=array.shape, dtype=array.dtype, buffer=array.copy()
        )

    def __hash__(self) -> int:
        if not hasattr(self, "_HashableNdarray__hash"):
            self.__hash = int(sha1(self.view()).hexdigest(), 16)
        return self.__hash

    def __eq__(self, other: Type[HashArray]) -> bool:
        if not isinstance(other, HashableNdarray):
            return super().__eq__(other)
        return super().__eq__(super(HashableNdarray, other)).all()
