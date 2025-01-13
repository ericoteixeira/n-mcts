from abc import ABC, abstractmethod
from typing import (
    List,
    Sequence,
    Callable,
    Optional,
    Union,
    Set
)
from qas.qml_models.qml_gate_ops import QMLPool, SUPPORTED_OPS_DICT
class StateOfMCTS(ABC):
    name:str

    @abstractmethod
    def __init__(self, current_k:List[int]=[], op_pool=None, maxDepth = 30, qubit_with_actions:Set = None, gate_limit_dict:Optional[dict] = None):
        self.max_depth = maxDepth
        self.current_k = current_k
        self.current_depth = len(self.current_k)
        assert self.current_depth <= self.max_depth
        self.pool_obj = op_pool
        self.op_name_dict = op_pool.pool
        self.pool_keys = list(op_pool.pool.keys())
        self.state = None
        self.qubit_with_actions = qubit_with_actions if qubit_with_actions is not None else set()
        self.gate_limit_dict = gate_limit_dict if gate_limit_dict is not None else {}
        self.gate_count = {}
        for key in self.gate_limit_dict.keys():
            self.gate_count[key] = 0
        for action in current_k:
            op = self.op_name_dict[action]
            assert len(op.keys()) == 1  # in case some weird things happen
            op_name = list(op.keys())[0]
            if op_name in set(self.gate_limit_dict.keys()):
                self.gate_count[op_name] = self.gate_count[op_name] + 1

    @abstractmethod
    def getLegalActions(self, *args):
        raise NotImplementedError
    @abstractmethod
    def takeAction(self,*args):
        raise NotImplementedError
    @abstractmethod
    def isTerminal(self):
        raise NotImplementedError
    @abstractmethod
    def getReward(self):
        raise NotImplementedError
    @abstractmethod
    def __repr__(self):
        raise NotImplementedError
    @abstractmethod
    def getCurrK(self):
        raise NotImplementedError


