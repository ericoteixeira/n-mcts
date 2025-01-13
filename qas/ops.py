from abc import ABC, abstractmethod
from typing import (
    List,
    Sequence,
    Any,
    Tuple,
    Callable,
    Iterator,
    Optional,
    Union,
    Iterable,
    Dict,
    AnyStr
)

class QuantumGate(ABC):
    @abstractmethod
    def __init__(self, *args):
        raise NotImplementedError
    @abstractmethod
    def __str__(self):
        raise NotImplementedError
    @abstractmethod
    def getOp(self):
        raise NotImplementedError

class Pool(dict):
    @abstractmethod
    def __init__(self, *args):
        super(Pool, self).__init__()
    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError
    @abstractmethod
    def __str__(self):
        raise NotImplementedError
    @abstractmethod
    def __len__(self):
        raise NotImplementedError