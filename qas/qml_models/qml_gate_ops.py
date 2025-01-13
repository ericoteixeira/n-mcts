import json
from pprint import pprint
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
import pennylane as qml
from pennylane.operation import Operation
from pennylane import (
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    S,
    T,
    SX,
    CNOT,
    CZ,
    CY,
    SWAP,
    ISWAP,
    SISWAP,
    SQISW,
    CSWAP,
    Toffoli,
    MultiControlledX,
    Rot,
    RX,
    RY,
    RZ,
    MultiRZ,
    PhaseShift,
    ControlledPhaseShift,
    CPhase,
    CRX,
    CRY,
    CRZ,
    CRot,
    U1,
    U2,
    U3,
    IsingXX,
    IsingYY,
    IsingZZ
)

class PlaceHolder(Operation):
    num_params = 0
    num_wires = 1
    par_domain = None
    @staticmethod
    def decomposition(wires):
        return [qml.Identity( wires=wires)]


SUPPORTED_OPS_DICT = {
    'Hadamard':Hadamard,
    'PauliX':PauliX,
    'PauliY':PauliY,
    'PauliZ':PauliZ,
    'S':S,
    'Sdg':qml.adjoint(S),
    'T':T,
    'Tdg':qml.adjoint(T),
    'SX':SX,
    'CNOT':CNOT,
    'CZ':CZ,
    'CY':CY,
    'SWAP':SWAP,
    'ISWAP':ISWAP,
    'SISWAP':SISWAP,
    'SQISW':SQISW,
    'CSWAP':CSWAP,
    'Toffoli':Toffoli,
    'MultiControlledX':MultiControlledX,
    'Rot':Rot,
    'RX':RX,
    'RY':RY,
    'RZ':RZ,
    'MultiRZ':MultiRZ,
    'PhaseShift':PhaseShift,
    'ControlledPhaseShift':ControlledPhaseShift,
    'CPhase':CPhase,
    'CRX':CRX,
    'CRY':CRY,
    'CRZ':CRZ,
    'CRot':CRot,
    'U1':U1,
    'U2':U2,
    'U3':U3,
    'IsingXX':IsingXX,
    'IsingYY':IsingYY,
    'IsingZZ':IsingZZ,
    'PlaceHolder':PlaceHolder
}

SUPPORTED_OPS_NAME = set(SUPPORTED_OPS_DICT.keys())

from qas.ops import Pool, QuantumGate

class QMLGate(QuantumGate):

    def __init__(self, name:str, pos:List[int], param:Optional[Sequence]=None):
        assert name in SUPPORTED_OPS_NAME
        op_constructor = SUPPORTED_OPS_DICT[name]
        self.num_params = op_constructor.num_params
        self.num_wires = op_constructor.num_wires
        if param is not None:
            assert len(param) == self.num_params
        assert len(pos) == self.num_wires
        self.wires = pos
        self.name = name
        self.op = op_constructor(*param, wires=self.wires) if param is not None else op_constructor(wires=self.wires)
        self.param = param if param is not None else None

    def getOp(self):
        return self.op

    def getQregPos(self):
        return list(self.wires)

    def __str__(self):
        disp = "{}, {}, {}".format(self.name, self.wires, list(self.param))
        return disp

class QMLPool(Pool):
    def __init__(self, num_qubits: int, single_qubit_op_generator: List[AnyStr],
                 two_qubit_op_generator: List[AnyStr],
                 complete_undirected_graph: bool = True, two_qubit_gate_map: Optional[List[List[int]]] = None):
        super(QMLPool, self).__init__()
        for c in single_qubit_op_generator:
            op_constructor = SUPPORTED_OPS_DICT[c]
            assert op_constructor.num_wires == 1
        for c in two_qubit_op_generator:
            op_constructor = SUPPORTED_OPS_DICT[c]
            assert op_constructor.num_wires == 2
        if two_qubit_gate_map is not None:
            assert complete_undirected_graph is False
            self.coupling = two_qubit_gate_map
            for c in self.coupling:
                assert len(c) == 2
                assert c[0] != c[1]
                assert c[0] >= 0
                assert c[0] < num_qubits
                assert c[1] >= 0
                assert c[1] < num_qubits
        else:
            self.coupling = []
            for i in range(num_qubits):
                for j in range(num_qubits):
                    if i != j:
                        self.coupling.append((i, j))

        self.num_qubits = num_qubits
        self.single_qubit_gates = single_qubit_op_generator
        self.two_qubit_gates = two_qubit_op_generator
        self.pool = {}
        pool_key = 0
        for i in range(num_qubits):
            for c in single_qubit_op_generator:
                self.pool[pool_key] = {c: [i]}
                pool_key = pool_key + 1
        for couple_direction in self.coupling:
            for c in two_qubit_op_generator:
                self.pool[pool_key] = {c: couple_direction}
                pool_key = pool_key + 1

    def __getitem__(self, key):
        return self.pool[key]

    def __str__(self):
        return json.dumps(self.pool)

    def __len__(self):
        return len(self.pool.keys())
    