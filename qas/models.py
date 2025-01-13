from abc import ABC, abstractmethod

class ModelFromK(ABC):
    name:str
    @abstractmethod
    def __init__(self, p:int, c:int, l:int, structure_list, op_pool, *args, **kargs):
        raise NotImplementedError
    @abstractmethod
    def getLoss(self, super_circ_params):
        raise NotImplementedError
    @abstractmethod
    def getGradient(self, super_circ_params):
        raise NotImplementedError
    @abstractmethod
    def toList(self, super_circ_params):
        # convert model to a list of (op_name, op_qubits, op_param)
        raise NotImplementedError
    @abstractmethod
    def getReward(self, super_circ_params):
        raise NotImplementedError

