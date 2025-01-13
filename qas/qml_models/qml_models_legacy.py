from qas.models import ModelFromK
from qas.qml_models.qml_gate_ops import (
    SUPPORTED_OPS_DICT,
    QMLPool,
    QMLGate
)
import pennylane.numpy as pnp
from pennylane import numpy as np
import pennylane as qml
from typing import (
    List,
    Sequence,
    Union
)

from collections import OrderedDict, Counter
import shutup
shutup.please()


ket0 = np.array([1, 0])
ket1 = np.array([0, 1])
PAULI_EIGENSTATES_T_STATE = [(ket0 + ket1) / np.sqrt(2), (ket0 - ket1) / np.sqrt(2), ket0, ket1,
                             (ket0 + ket1 * 1j) / np.sqrt(2), (ket0 - ket1 * 1j) / np.sqrt(2),
                             (ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)]
THREE_QUBIT_BASIS = {
    '000':np.kron(ket0, np.kron(ket0, ket0)),
    '001':np.kron(ket0, np.kron(ket0, ket1)),
    '010':np.kron(ket0, np.kron(ket1, ket0)),
    '011':np.kron(ket0, np.kron(ket1, ket1)),
    '100':np.kron(ket1, np.kron(ket0, ket0)),
    '101':np.kron(ket1, np.kron(ket0, ket1)),
    '110':np.kron(ket1, np.kron(ket1, ket0)),
    '111':np.kron(ket1, np.kron(ket1, ket1))
}

PAULI_X = np.array([[0,1],[1,0]])
PAULI_Z = np.array(([[1,0],[0,-1]]))
PAULI_Y = np.array([[0,-1j],[1j,0]])
IDENTITY = np.array([[1,0],[0,1]])

STATE_DIC = {'|0>': ket0, '|1>': ket1, '|+>': (ket0 + ket1) / np.sqrt(2), '|->': (ket0 - ket1) / np.sqrt(2),
             '|+i>': (ket0 + ket1 * 1j) / np.sqrt(2), '|-i>': (ket0 - ket1 * 1j) / np.sqrt(2),
             'magic': (ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)}
STATE_DICT_KEYS = list(STATE_DIC.keys())

FOUR_TWO_TWO_DETECTION_CODE_INPUT = []
for c in PAULI_EIGENSTATES_T_STATE:
    for d in PAULI_EIGENSTATES_T_STATE:
        FOUR_TWO_TWO_DETECTION_CODE_INPUT.append(np.kron(c,d))

TOFFOLI_INPUT = []
EXTENDED_TOFFILI_INPUT = []
FULL_TOFFOLI_INPUT = []
_de = 1/np.sqrt(8) #0.354
# (qubit 0,1,2:) STZ
EXTENDED_TOFFILI_INPUT.append(np.array([ _de+0j, 0+_de*1j, 0.25+0.25j, -0.25+0.25j, -_de+0j, 0-_de*1j, -0.25-0.25j, 0.25-0.25j ]))
# T^dg, S^dg, Sqrt(X)
EXTENDED_TOFFILI_INPUT.append(np.array([ _de+0j, 0.25-0.25j, 0-_de*1j, -0.25-0.25j, _de+0j, 0.25-0.25j, 0-_de*1j, -0.25-0.25j ]))
# Y, Z, Sqrt(X)
EXTENDED_TOFFILI_INPUT.append(np.array([ 0-_de*1j, 0+_de*1j, 0+_de*1j, 0-_de*1j, 0-_de*1j, 0+_de*1j, 0+_de*1j, 0-_de*1j ]))
# P, Z, Y
EXTENDED_TOFFILI_INPUT.append(np.array([ 0-_de*1j, _de+0j, 0+_de*1j, -_de+0j, 0+_de*1j, -_de+0j, 0-_de*1j, _de+0j ]))

# \ket{000}+\ket{001}+\ket{010}+\ket{011}+\ket{100}+\ket{101}+\ket{110}-\ket{111}
EXTENDED_TOFFILI_INPUT.append(1/np.sqrt(8)*(THREE_QUBIT_BASIS['000']+THREE_QUBIT_BASIS['001']+
                                            THREE_QUBIT_BASIS['010']+THREE_QUBIT_BASIS['011']+
                                            THREE_QUBIT_BASIS['100']+THREE_QUBIT_BASIS['101']+
                                            THREE_QUBIT_BASIS['110']-THREE_QUBIT_BASIS['111']))

FULL_TOFFOLI_INPUT.extend(EXTENDED_TOFFILI_INPUT)
for a in [ket0, ket1]:
    for b in [ket0, ket1]:
        for c in [ket0, ket1]:
            temp = np.kron(a, b)
            s = np.kron(temp, c)
            TOFFOLI_INPUT.append(s)
for a in [ket0, ket1,(ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)]:
    for b in [ket0, ket1,(ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)]:
        for c in [ket0, ket1,(ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)]:
            temp = np.kron(a, b)
            s = np.kron(temp, c)
            EXTENDED_TOFFILI_INPUT.append(s)

for a in PAULI_EIGENSTATES_T_STATE:
    for b in PAULI_EIGENSTATES_T_STATE:
        for c in PAULI_EIGENSTATES_T_STATE:
            temp = np.kron(a, b)
            s = np.kron(temp, c)
            FULL_TOFFOLI_INPUT.append(s)
TWO_QUBIT_ENTANGLED_STATES = []
TWO_QUBIT_ENTANGLED_STATES.append(1/np.sqrt(2)*(np.kron(ket0, ket0)+np.kron(ket1, ket1)))
TWO_QUBIT_ENTANGLED_STATES.append(1/np.sqrt(2)*(np.kron(ket0, ket0)-np.kron(ket1, ket1)))
TWO_QUBIT_ENTANGLED_STATES.append(1/np.sqrt(2)*(np.kron(ket1, ket0)+np.kron(ket0, ket1)))
TWO_QUBIT_ENTANGLED_STATES.append(1/np.sqrt(2)*(np.kron(ket1, ket0)-np.kron(ket0, ket1)))
TWO_QUBIT_ENTANGLED_STATES.append(1/np.sqrt(2)*(np.kron(ket0, ket0)+1j*np.kron(ket1, ket1)))
TWO_QUBIT_ENTANGLED_STATES.append(1/np.sqrt(2)*(np.kron(ket0, ket0)-1j*np.kron(ket1, ket1)))
TWO_QUBIT_ENTANGLED_STATES.append(1/np.sqrt(2)*(np.kron(ket1, ket0)+1j*np.kron(ket0, ket1)))
TWO_QUBIT_ENTANGLED_STATES.append(1/np.sqrt(2)*(np.kron(ket1, ket0)-1j*np.kron(ket0, ket1)))
for c in TWO_QUBIT_ENTANGLED_STATES:
    for d in PAULI_EIGENSTATES_T_STATE:
        s = np.kron(c, d)
        s1 = np.kron(d, c)
        FULL_TOFFOLI_INPUT.append(s)
        FULL_TOFFOLI_INPUT.append(s1)

THREE_QUBIT_GHZ_STATES = []
THREE_QUBIT_GHZ_STATES.append(1/np.sqrt(2)*(np.kron(ket0, np.kron(ket0, ket0))+np.kron(ket1, np.kron(ket1, ket1))))
THREE_QUBIT_GHZ_STATES.append(1/np.sqrt(2)*(np.kron(ket0, np.kron(ket0, ket0))-np.kron(ket1, np.kron(ket1, ket1))))
THREE_QUBIT_GHZ_STATES.append(1/np.sqrt(2)*(np.kron(ket0, np.kron(ket0, ket0))+1j*np.kron(ket1, np.kron(ket1, ket1))))
THREE_QUBIT_GHZ_STATES.append(1/np.sqrt(2)*(np.kron(ket0, np.kron(ket0, ket0))-1j*np.kron(ket1, np.kron(ket1, ket1))))
EXTENDED_TOFFILI_INPUT.extend(THREE_QUBIT_GHZ_STATES)
FOUR_TWO_TWO_DETECTION_CODE_INPUT.extend(TWO_QUBIT_ENTANGLED_STATES)
FULL_TOFFOLI_INPUT.extend(THREE_QUBIT_GHZ_STATES)

SIMPLE_PHASE_FLIP_DATA = []
for x in PAULI_EIGENSTATES_T_STATE:
    dev_pf = qml.device('default.qubit', wires=3)
    @qml.qnode(dev_pf)
    def phase_flip_circ(x):
        qml.QubitStateVector(x, wires=[0])
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0,1])
        qml.CNOT(wires=[0,2])
        return qml.density_matrix(wires=[0,1,2])
    SIMPLE_PHASE_FLIP_DATA.append((x, phase_flip_circ(x)))

TOFFOLI_DATA = []
for x in TOFFOLI_INPUT:
    dev_t = qml.device('default.qubit', wires=3)
    @qml.qnode(dev_t)
    def toffoli_circ(x):
        qml.QubitStateVector(x, wires=[0,1,2])
        qml.Toffoli(wires=[0,1,2])
        return qml.density_matrix(wires=[0, 1, 2])
    TOFFOLI_DATA.append((x, toffoli_circ(x)))

EXTENDED_TOFFOLI_DATA = []
for x in EXTENDED_TOFFILI_INPUT:
    dev_t = qml.device('default.qubit', wires=3)
    @qml.qnode(dev_t)
    def toffoli_circ(x):
        qml.QubitStateVector(x, wires=[0, 1, 2])
        qml.Toffoli(wires=[0, 1, 2])
        return qml.density_matrix(wires=[0, 1, 2])
    EXTENDED_TOFFOLI_DATA.append((x, toffoli_circ(x)))

FULL_TOFFOLI_DATA = []
for x in FULL_TOFFOLI_INPUT:
    dev_t = qml.device('default.qubit', wires=3)
    @qml.qnode(dev_t)
    def toffoli_circ(x):
        qml.QubitStateVector(x, wires=[0, 1, 2])
        qml.Toffoli(wires=[0, 1, 2])
        return qml.density_matrix(wires=[0, 1, 2])
    FULL_TOFFOLI_DATA.append((x, toffoli_circ(x)))

FOUR_TWO_TWO_DETECTION_CODE_DATA = []
for x in FOUR_TWO_TWO_DETECTION_CODE_INPUT:
    dev_422 = qml.device('default.qubit', wires=4)
    @qml.qnode(dev_422)
    def four_two_two_circ(x):
        qml.QubitStateVector(x, wires=[0, 1])
        qml.Hadamard(wires=[3])
        qml.CNOT(wires=[0,2])
        qml.CNOT(wires=[1,2])
        qml.CNOT(wires=[3,2])
        qml.CNOT(wires=[3,1])
        qml.CNOT(wires=[3,0])
        return qml.density_matrix(wires=[0,1,2,3])
    FOUR_TWO_TWO_DETECTION_CODE_DATA.append((x, four_two_two_circ(x)))

FIVE_ONE_THREE_QECC_DATA = []
for x in [ket0, ket1]:
    dev_513 = qml.device('default.qubit', wires = 5)
    @qml.qnode(dev_513)
    def five_one_three_circ(x):
        qml.QubitStateVector(x, wires=[0])
        qml.PauliZ(wires=[0])
        qml.Hadamard(wires=[2])
        qml.Hadamard(wires=[3])
        qml.adjoint(qml.S)(wires=[0])
        qml.CNOT(wires=[2,4])
        qml.CNOT(wires=[3,1])
        qml.Hadamard(wires=[1])
        qml.CNOT(wires=[3,4])
        qml.CNOT(wires=[1,0])
        qml.adjoint(qml.S)(wires=[2])
        qml.S(wires=[3])
        qml.adjoint(qml.S)(wires=4)
        qml.S(wires=0)
        qml.S(wires=1)
        qml.PauliZ(wires=2)
        qml.CNOT(wires=[4,0])
        qml.Hadamard(wires=4)
        qml.CNOT(wires=[4,1])
        return qml.density_matrix(wires=[0,1,2,3,4])
    FIVE_ONE_THREE_QECC_DATA.append((x, five_one_three_circ(x)))


def extractParamIndicesQML(k:List[int], op_pool:Union[QMLPool, dict])->List:
    assert min(k) >= 0
    p = len(k)
    c = len(op_pool)
    assert max(k) < c
    param_indices = []
    for i in range(p):
        gate_dict = op_pool[k[i]]
        assert len(gate_dict.keys()) == 1
        gate_name = list(gate_dict.keys())[0]
        gate_obj = SUPPORTED_OPS_DICT[gate_name]
        num_params = gate_obj.num_params
        if num_params > 0:
            for j in range(num_params):
                param_indices.append((i, k[i], j))
    return param_indices

def generate_single_qubit_state(theta_B:float, phi_B:float)->np.ndarray:
    return np.cos(theta_B/2)*ket0 + np.sin(theta_B/2)*np.exp(1j*phi_B)*ket1

class PhaseFlipQMLNoiseless(ModelFromK):
    name = "PhaseFlipQMLNoiseless"
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 3
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.data = SIMPLE_PHASE_FLIP_DATA
        self.dev = qml.device('default.qubit', wires=3)

    #@qml.template
    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name !="PlaceHolder":
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                num_params = gate_obj.num_params
                wires = gate_dict[gate_name]
                if num_params > 0:
                    gate_params = []
                    for j in range(num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                else:
                    gate_params = None
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                qml_gate_obj.getOp()

    def constructFullCirc(self):
        @qml.qnode(self.dev)
        def fullCirc(extracted_params, x=None, y = None):
            qml.QubitStateVector(x, wires=[0])
            self.backboneCirc(extracted_params)
            return qml.expval(qml.Hermitian(y, wires=[0,1,2]))
        return fullCirc

    def costFunc(self, extracted_params):
        fid = 0
        circ_func = self.constructFullCirc()
        num_data = len(self.data)
        for i in range(num_data):
            fid = fid + circ_func(extracted_params, x=self.data[i][0], y=self.data[i][1])
        return 1 - fid/num_data

    def getLoss(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return self.costFunc(extracted_params)

    def getReward(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return 1-self.costFunc(extracted_params)


    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.costFunc)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]

        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_pos = gate_dict[gate_name]
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                gate_num_params = gate_obj.num_params
                if gate_num_params > 0:
                    gate_params = []
                    for j in range(gate_num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    gate_list.append((gate_name, gate_pos, gate_params))
                else:
                    gate_list.append((gate_name, gate_pos, None))

        return gate_list

class ToffoliQMLNoiseless(ModelFromK):
    name = "ToffoliQMLNoiseless"
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 3
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.data = TOFFOLI_DATA
        self.dev = qml.device('default.qubit', wires = self.num_qubits)
        #self.dev = qml.device('qiskit.aer', wires=self.num_qubits, max_parallel_threads=0, max_parallel_experiments=0)

    #@qml.template
    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            gate_obj = SUPPORTED_OPS_DICT[gate_name]
            num_params = gate_obj.num_params
            wires = gate_dict[gate_name]
            if num_params > 0:
                gate_params = []
                for j in range(num_params):
                    gate_params.append(extracted_params[param_pos])
                    param_pos = param_pos + 1
                qml_gate_obj = QMLGate(gate_name, wires, gate_params)
            else:
                gate_params = None
                qml_gate_obj = QMLGate(gate_name, wires, gate_params)
            qml_gate_obj.getOp()

    def constructFullCirc(self):
        @qml.qnode(self.dev)
        def fullCirc(extracted_params, x=None, y = None):
            qml.QubitStateVector(x, wires=[0,1,2])
            self.backboneCirc(extracted_params)
            return qml.expval(qml.Hermitian(y, wires=[0,1,2]))
        return fullCirc

    def costFunc(self, extracted_params):
        fid = 0
        circ_func = self.constructFullCirc()
        num_data = len(self.data)
        for c in self.data:
            fid = fid + circ_func(extracted_params, x=c[0], y=c[1])
        return 1 - fid/num_data

    def getLoss(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return self.costFunc(extracted_params)

    def getReward(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return 1-self.costFunc(extracted_params)


    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.costFunc)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]

        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            gate_pos = gate_dict[gate_name]
            gate_obj = SUPPORTED_OPS_DICT[gate_name]
            gate_num_params = gate_obj.num_params
            if gate_num_params > 0:
                gate_params = []
                for j in range(gate_num_params):
                    gate_params.append(extracted_params[param_pos])
                    param_pos = param_pos + 1
                gate_list.append((gate_name, gate_pos, gate_params))
            else:
                gate_list.append((gate_name, gate_pos, None))

        return gate_list

class ToffoliQMLNoiselessAdditionalData(ModelFromK):
    name = "ToffoliQMLNoiselessAdditionalData"
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 3
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.data = EXTENDED_TOFFOLI_DATA
        self.dev = qml.device('default.qubit', wires = self.num_qubits)
        #self.dev = qml.device('qiskit.aer', wires=self.num_qubits, max_parallel_threads=0, max_parallel_experiments=0)

    #@qml.template
    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            gate_obj = SUPPORTED_OPS_DICT[gate_name]
            num_params = gate_obj.num_params
            wires = gate_dict[gate_name]
            if num_params > 0:
                gate_params = []
                for j in range(num_params):
                    gate_params.append(extracted_params[param_pos])
                    param_pos = param_pos + 1
                qml_gate_obj = QMLGate(gate_name, wires, gate_params)
            else:
                gate_params = None
                qml_gate_obj = QMLGate(gate_name, wires, gate_params)
            qml_gate_obj.getOp()

    def constructFullCirc(self):
        @qml.qnode(self.dev)
        def fullCirc(extracted_params, x=None, y = None):
            qml.QubitStateVector(x, wires=[0,1,2])
            self.backboneCirc(extracted_params)
            return qml.expval(qml.Hermitian(y, wires=[0,1,2]))
        return fullCirc

    def costFunc(self, extracted_params):
        fid = 0
        circ_func = self.constructFullCirc()
        num_data = len(self.data)
        for c in self.data:
            fid = fid + circ_func(extracted_params, x=c[0], y=c[1])
        return 1 - fid/num_data

    def getLoss(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return self.costFunc(extracted_params)

    def getReward(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return 1-self.costFunc(extracted_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.costFunc)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]

        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            gate_pos = gate_dict[gate_name]
            gate_obj = SUPPORTED_OPS_DICT[gate_name]
            gate_num_params = gate_obj.num_params
            if gate_num_params > 0:
                gate_params = []
                for j in range(gate_num_params):
                    gate_params.append(extracted_params[param_pos])
                    param_pos = param_pos + 1
                gate_list.append((gate_name, gate_pos, gate_params))
            else:
                gate_list.append((gate_name, gate_pos, None))

        return gate_list

class ToffoliQMLNoiselessFullInput(ModelFromK):
    name = "ToffoliQMLNoiselessFullInput"
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 3
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.data = FULL_TOFFOLI_DATA
        self.dev = qml.device('default.qubit', wires = self.num_qubits)
        #self.dev = qml.device('qiskit.aer', wires=self.num_qubits, max_parallel_threads=0, max_parallel_experiments=0)

    #@qml.template
    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            gate_obj = SUPPORTED_OPS_DICT[gate_name]
            num_params = gate_obj.num_params
            wires = gate_dict[gate_name]
            if num_params > 0:
                gate_params = []
                for j in range(num_params):
                    gate_params.append(extracted_params[param_pos])
                    param_pos = param_pos + 1
                qml_gate_obj = QMLGate(gate_name, wires, gate_params)
            else:
                gate_params = None
                qml_gate_obj = QMLGate(gate_name, wires, gate_params)
            qml_gate_obj.getOp()

    def constructFullCirc(self):
        @qml.qnode(self.dev)
        def fullCirc(extracted_params, x=None, y = None):
            qml.QubitStateVector(x, wires=[0,1,2])
            self.backboneCirc(extracted_params)
            return qml.expval(qml.Hermitian(y, wires=[0,1,2]))
        return fullCirc

    def costFunc(self, extracted_params):
        fid = 0
        circ_func = self.constructFullCirc()
        num_data = len(self.data)
        for c in self.data:
            fid = fid + circ_func(extracted_params, x=c[0], y=c[1])
        return 1 - fid/num_data

    def getLoss(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return self.costFunc(extracted_params)

    def getReward(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return 1-self.costFunc(extracted_params)


    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.costFunc)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]

        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            gate_pos = gate_dict[gate_name]
            gate_obj = SUPPORTED_OPS_DICT[gate_name]
            gate_num_params = gate_obj.num_params
            if gate_num_params > 0:
                gate_params = []
                for j in range(gate_num_params):
                    gate_params.append(extracted_params[param_pos])
                    param_pos = param_pos + 1
                gate_list.append((gate_name, gate_pos, gate_params))
            else:
                gate_list.append((gate_name, gate_pos, None))

        return gate_list

class ToffoliQMLSwapTestNoiselessExtendedData(ModelFromK):
    name = "ToffoliQMLSwapTestNoiselessExtendedData"
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 3
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.x_list = EXTENDED_TOFFILI_INPUT
        self.dev = qml.device('default.qubit', wires = self.num_qubits*2+1)
        #self.dev = qml.device('qiskit.aer', wires=self.num_qubits, max_parallel_threads=0, max_parallel_experiments=0)

    #@qml.template
    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            gate_obj = SUPPORTED_OPS_DICT[gate_name]
            num_params = gate_obj.num_params
            wires = gate_dict[gate_name]
            if num_params > 0:
                gate_params = []
                for j in range(num_params):
                    gate_params.append(extracted_params[param_pos])
                    param_pos = param_pos + 1
                qml_gate_obj = QMLGate(gate_name, wires, gate_params)
            else:
                gate_params = None
                qml_gate_obj = QMLGate(gate_name, wires, gate_params)
            qml_gate_obj.getOp()

    def constructFullCirc(self):
        @qml.qnode(self.dev)
        def fullCirc(extracted_params, x=None):

            qml.QubitStateVector(np.kron(x,x), wires=[0,1,2,3,4,5])

            self.backboneCirc(extracted_params)
            qml.Toffoli(wires=[3,4,5])
            qml.Hadamard(wires=6)
            qml.CSWAP(wires=[6,5,2])
            qml.CSWAP(wires=[6,4,1])
            qml.CSWAP(wires=[6,3,0])
            qml.Hadamard(wires=6)
            return qml.expval(qml.Hermitian(np.outer(ket0, ket0), wires=6))
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        fid = 0
        for i in range(len(self.x_list)):
            fid = fid +  (circ_func(extracted_params, x=self.x_list[i])-1/2)*2
        return 1 - fid/len(self.x_list)

    def getLoss(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return self.costFunc(extracted_params)

    def getReward(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return 1-self.costFunc(extracted_params)


    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.costFunc)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]

        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            gate_pos = gate_dict[gate_name]
            gate_obj = SUPPORTED_OPS_DICT[gate_name]
            gate_num_params = gate_obj.num_params
            if gate_num_params > 0:
                gate_params = []
                for j in range(gate_num_params):
                    gate_params.append(extracted_params[param_pos])
                    param_pos = param_pos + 1
                gate_list.append((gate_name, gate_pos, gate_params))
            else:
                gate_list.append((gate_name, gate_pos, None))

        return gate_list

class FourTwoTwoQMLNoiseless(ModelFromK):
    name="FourTwoTwoQMLNoiseless"
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 4
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.data = FOUR_TWO_TWO_DETECTION_CODE_DATA
        self.dev = qml.device('default.qubit', wires = self.num_qubits)

    #@qml.template
    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                num_params = gate_obj.num_params
                wires = gate_dict[gate_name]
                if num_params > 0:
                    gate_params = []
                    for j in range(num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                else:
                    gate_params = None
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                qml_gate_obj.getOp()

    def constructFullCirc(self):
        @qml.qnode(self.dev)
        def fullCirc(extracted_params, x=None, y = None):
            qml.QubitStateVector(x, wires=[0,1])
            self.backboneCirc(extracted_params)
            return qml.expval(qml.Hermitian(y, wires=[0,1,2,3]))
        return fullCirc

    def costFunc(self, extracted_params):
        fid = 0
        circ_func = self.constructFullCirc()
        num_data = len(self.data)
        for i in range(num_data):
            fid = fid + circ_func(extracted_params, x=self.data[i][0], y=self.data[i][1])
        return 1-fid/num_data

    def getLoss(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return self.costFunc(extracted_params)

    def getReward(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return 1-self.costFunc(extracted_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        #cost_grad = jax.grad(self.costFunc, argnums=0)
        cost_grad = qml.grad(self.costFunc)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]

        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_pos = gate_dict[gate_name]
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                gate_num_params = gate_obj.num_params
                if gate_num_params > 0:
                    gate_params = []
                    for j in range(gate_num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    gate_list.append((gate_name, gate_pos, gate_params))
                else:
                    gate_list.append((gate_name, gate_pos, None))

        return gate_list

class PrepareLogicalKetZeroState513QECC(ModelFromK):
    name = "PrepareLogicalState513QECC_LogicalKetZero"
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 5
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('default.qubit', wires = self.num_qubits)
        self.logical_0_vector = np.array([1/4, 0, 0, 1/4, 0, -(1/4), 1/4, 0, 0, -(1/4), -(1/4), 0, 1/4, 0, 0, -(1/4), 0, 1/4, -(1/4), 0, -(1/4), 0, 0, -(1/4), 1/4, 0, 0, -(1/4), 0, -(1/4), -(1/4), 0])# https://github.com/bernwo/five-qubit-code/blob/95ff95a3933baaa95ea3649cc28a2f1da4175af3/main.py#L20
        self.logical_1_vector = np.array([0, -(1/4), -(1/4), 0, -(1/4), 0, 0, 1/4, -(1/4), 0, 0, -(1/4), 0, -(1/4), 1/4, 0, -(1/4), 0, 0, 1/4, 0, -(1/4), -(1/4), 0, 0, 1/4, -(1/4), 0, 1/4, 0, 0, 1/4])# https://github.com/bernwo/five-qubit-code/blob/95ff95a3933baaa95ea3649cc28a2f1da4175af3/main.py#L32
        self.logical_0_observable = qml.Hermitian(np.outer(self.logical_0_vector, self.logical_0_vector),wires=[0,1,2,3,4])
        #self.observable = qml.Hermitian(-1/5*(self.g1+self.g2+self.g3+self.g4+self.O), wires=[0,1,2,3,4])

    #@qml.template
    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                num_params = gate_obj.num_params
                wires = gate_dict[gate_name]
                if num_params > 0:
                    gate_params = []
                    for j in range(num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                else:
                    gate_params = None
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                qml_gate_obj.getOp()

    def constructFullCirc(self):
        @qml.qnode(self.dev)
        def fullCirc(extracted_params):
            self.backboneCirc(extracted_params)
            return qml.expval(self.logical_0_observable)
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        overlap = circ_func(extracted_params)
        return 1-overlap

    def getLoss(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        cost = self.costFunc(extracted_params)
        return cost

    def getReward(self, super_circ_params):
        return 1-self.getLoss(super_circ_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        #print(len(extracted_params))
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.costFunc)
        #extracted_params = np.array(extracted_params, requires_grad=True) # needed for the new pennylane version
        extracted_gradients = cost_grad(extracted_params)
        #print(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]
        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_pos = gate_dict[gate_name]
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                gate_num_params = gate_obj.num_params
                if gate_num_params > 0:
                    gate_params = []
                    for j in range(gate_num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    gate_list.append((gate_name, gate_pos, gate_params))
                else:
                    gate_list.append((gate_name, gate_pos, None))

        return gate_list

class PrepareLogicalKetMinusState513QECC(ModelFromK):
    coeff_dict = {
        'ket0':{'alpha':1, 'beta':0},
        'ket1':{'alpha':0, 'beta':1},
        'ket_plus':{'alpha':STATE_DIC['|+>'][0], 'beta':STATE_DIC['|+>'][1]},
        'ket_minus':{'alpha':STATE_DIC['|->'][0], 'beta':STATE_DIC['|->'][1]},
        'ket_plus_i':{'alpha':STATE_DIC['|+i>'][0], 'beta':STATE_DIC['|+i>'][1]},
        'ket_minus_i':{'alpha':STATE_DIC['|-i>'][0], 'beta':STATE_DIC['|-i>'][1]},
        'ket_T_state':{'alpha':STATE_DIC['magic'][0], 'beta':STATE_DIC['magic'][1]}
    }
    target_state_name = 'ket_minus'
    name = "PrepareLogicalKetMinusState513QECC"
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 5
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('default.qubit', wires = self.num_qubits)
        self.state_alpha = self.coeff_dict[self.target_state_name]['alpha']
        self.state_beta = self.coeff_dict[self.target_state_name]['beta']
        self.logical_X = np.kron(np.kron(np.kron(PAULI_X,np.kron(PAULI_X,PAULI_X)),PAULI_X),PAULI_X)
        self.logical_Y = np.kron(np.kron(np.kron(PAULI_Y,np.kron(PAULI_Y,PAULI_Y)),PAULI_Y),PAULI_Y)
        self.logical_Z = np.kron(np.kron(np.kron(PAULI_Z,np.kron(PAULI_Z,PAULI_Z)),PAULI_Z),PAULI_Z)
        self.O = (self.state_alpha*np.conj(self.state_beta)+np.conj(self.state_alpha)*self.state_beta)*self.logical_X-1j*(np.conj(self.state_alpha)*self.state_beta-self.state_alpha*np.conj(self.state_beta))*self.logical_Y-(self.state_beta*np.conj(self.state_beta)-self.state_alpha*np.conj(self.state_alpha))*self.logical_Z
        self.g1 = np.kron(np.kron(np.kron(np.kron(PAULI_X, PAULI_Z), PAULI_Z), PAULI_X), IDENTITY)
        self.g2 = np.kron(np.kron(np.kron(np.kron(IDENTITY, PAULI_X),PAULI_Z),PAULI_Z),PAULI_X)
        self.g3 = np.kron(np.kron(np.kron(np.kron(PAULI_X, IDENTITY), PAULI_X), PAULI_Z), PAULI_Z)
        self.g4 = np.kron(np.kron(np.kron(np.kron(PAULI_Z, PAULI_X), IDENTITY), PAULI_X), PAULI_Z)


        self.observable = qml.Hermitian(-1/5*(self.g1+self.g2+self.g3+self.g4+self.O), wires=[0,1,2,3,4])

    #@qml.template
    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                num_params = gate_obj.num_params
                wires = gate_dict[gate_name]
                if num_params > 0:
                    gate_params = []
                    for j in range(num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                else:
                    gate_params = None
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                qml_gate_obj.getOp()

    def constructFullCirc(self):
        @qml.qnode(self.dev)
        def fullCirc(extracted_params):
            self.backboneCirc(extracted_params)
            return qml.expval(self.observable)
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        energy = circ_func(extracted_params)
        return energy

    def getLoss(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        cost = self.costFunc(extracted_params)
        return cost

    def getReward(self, super_circ_params):
        return -self.getLoss(super_circ_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.costFunc)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]
        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_pos = gate_dict[gate_name]
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                gate_num_params = gate_obj.num_params
                if gate_num_params > 0:
                    gate_params = []
                    for j in range(gate_num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    gate_list.append((gate_name, gate_pos, gate_params))
                else:
                    gate_list.append((gate_name, gate_pos, None))

        return gate_list

class FiveOneThreeQECCNoiseless(ModelFromK):
    name = 'FiveOneThreeQECCNoiseless'
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 5
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.data = FIVE_ONE_THREE_QECC_DATA
        self.dev = qml.device('default.qubit', wires=5)

    #@qml.template
    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                num_params = gate_obj.num_params
                wires = gate_dict[gate_name]
                if num_params > 0:
                    gate_params = []
                    for j in range(num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                else:
                    gate_params = None
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                qml_gate_obj.getOp()

    def constructFullCirc(self):
        @qml.qnode(self.dev)
        def fullCirc(extracted_params, x=None, y = None):
            qml.QubitStateVector(x, wires=[0])
            self.backboneCirc(extracted_params)
            return qml.expval(qml.Hermitian(y, wires=[0,1,2,3,4]))
        return fullCirc

    def costFunc(self, extracted_params):
        fid = 0
        circ_func = self.constructFullCirc()
        num_data = len(self.data)
        for i in range(num_data):
            fid = fid + circ_func(extracted_params, x=self.data[i][0], y=self.data[i][1])
        return 1 - fid/num_data

    def getLoss(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return self.costFunc(extracted_params)

    def getReward(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return 1-self.costFunc(extracted_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.costFunc)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]

        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_pos = gate_dict[gate_name]
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                gate_num_params = gate_obj.num_params
                if gate_num_params > 0:
                    gate_params = []
                    for j in range(gate_num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    gate_list.append((gate_name, gate_pos, gate_params))
                else:
                    gate_list.append((gate_name, gate_pos, None))

        return gate_list

_H2_SYMBOLS = ["H", "H"]
_H2_COORDINATES = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
_H2_HAM_FOUR_QUBIT, _ = qml.qchem.molecular_hamiltonian(_H2_SYMBOLS, _H2_COORDINATES) #ground-state energy = -1.13618883 Ha

_LiH_SYMBOLS = ["Li", "H"]
_LiH_COORDINATES = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.969280527]) # units in Bohr
_LiH_HAM, _LiH_QUBITS = qml.qchem.molecular_hamiltonian(_LiH_SYMBOLS, _LiH_COORDINATES, active_electrons=2, active_orbitals=5, basis='sto-6g', load_data=True)

_H2O_SYMBOLS = ['H', 'O', 'H']
_H2O_COORDINATES = np.array([0.,0.,0.,1.63234543, 0.86417176, 0., 3.36087791, 0.,0.]) *1.88973 # Angstrom to Bohr
_H2O_HAM, _H2O_QUBITS = qml.qchem.molecular_hamiltonian(
    _H2O_SYMBOLS,
    _H2O_COORDINATES,
    charge=0,
    mult=1,
    basis="sto-6g",
    active_electrons=4,
    active_orbitals=4,
    load_data=True
)

class FourQubitH2(ModelFromK):

    name = "FourQubitH2"

    def __init__(self, p: int, c: int, l: int, structure_list: List[int], op_pool: Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 4
        self.num_electrons = 2
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('default.qubit', wires=self.num_qubits)
        self.H= _H2_HAM_FOUR_QUBIT  # ground-state energy = -1.13618883 Ha
        self.hf = qml.qchem.hf_state(self.num_electrons, self.num_qubits)

    #@qml.template
    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                num_params = gate_obj.num_params
                wires = gate_dict[gate_name]
                if num_params > 0:
                    gate_params = []
                    for j in range(num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                else:
                    gate_params = None
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                qml_gate_obj.getOp()

    def constructFullCirc(self):
        @qml.qnode(self.dev)
        def fullCirc(extracted_params):
            qml.BasisState(self.hf, wires=[0,1,2,3])
            self.backboneCirc(extracted_params)
            return qml.expval(self.H)
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        return circ_func(extracted_params)

    def getLoss(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return self.costFunc(extracted_params)

    def getReward(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return -self.costFunc(extracted_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.costFunc)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]

        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_pos = gate_dict[gate_name]
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                gate_num_params = gate_obj.num_params
                if gate_num_params > 0:
                    gate_params = []
                    for j in range(gate_num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    gate_list.append((gate_name, gate_pos, gate_params))
                else:
                    gate_list.append((gate_name, gate_pos, None))

        return gate_list

class FourQubitH2_VaccumInitial(ModelFromK):

    name = "FourQubitH2_VaccumInitial"

    def __init__(self, p: int, c: int, l: int, structure_list: List[int], op_pool: Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 4
        self.num_electrons = 2
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('default.qubit', wires=self.num_qubits)
        self.H= _H2_HAM_FOUR_QUBIT  # ground-state energy = -1.13618883 Ha
        self.hf = qml.qchem.hf_state(self.num_electrons, self.num_qubits)

    #@qml.template
    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                num_params = gate_obj.num_params
                wires = gate_dict[gate_name]
                if num_params > 0:
                    gate_params = []
                    for j in range(num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                else:
                    gate_params = None
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                qml_gate_obj.getOp()

    def constructFullCirc(self):
        @qml.qnode(self.dev)
        def fullCirc(extracted_params):
            #qml.BasisState(self.hf, wires=[0,1,2,3])
            self.backboneCirc(extracted_params)
            return qml.expval(self.H)
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        return circ_func(extracted_params)

    def getLoss(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return self.costFunc(extracted_params)

    def getReward(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return -self.costFunc(extracted_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.costFunc)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]

        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_pos = gate_dict[gate_name]
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                gate_num_params = gate_obj.num_params
                if gate_num_params > 0:
                    gate_params = []
                    for j in range(gate_num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    gate_list.append((gate_name, gate_pos, gate_params))
                else:
                    gate_list.append((gate_name, gate_pos, None))

        return gate_list

class LiH(ModelFromK):

    name = "LiH"

    def __init__(self, p: int, c: int, l: int, structure_list: List[int], op_pool: Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = _LiH_QUBITS
        self.num_active_electrons = 2
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('default.qubit', wires=self.num_qubits)
        self.H= qml.utils.sparse_hamiltonian(_LiH_HAM)  # ground-state energy = -7.8825378193 Ha
        self.hf = qml.qchem.hf_state(self.num_active_electrons, self.num_qubits)

    #@qml.template
    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                num_params = gate_obj.num_params
                wires = gate_dict[gate_name]
                if num_params > 0:
                    gate_params = []
                    for j in range(num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                else:
                    gate_params = None
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                qml_gate_obj.getOp()

    def constructFullCirc(self):
        @qml.qnode(self.dev,diff_method="parameter-shift")
        def fullCirc(extracted_params):
            #qml.BasisState(self.hf, wires=range(self.num_qubits))
            self.backboneCirc(extracted_params)
            return qml.expval(qml.SparseHamiltonian(self.H, wires=range(self.num_qubits)))
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        return circ_func(extracted_params)

    def getLoss(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return self.costFunc(extracted_params)

    def getReward(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return -self.costFunc(extracted_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.costFunc)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]

        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_pos = gate_dict[gate_name]
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                gate_num_params = gate_obj.num_params
                if gate_num_params > 0:
                    gate_params = []
                    for j in range(gate_num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    gate_list.append((gate_name, gate_pos, gate_params))
                else:
                    gate_list.append((gate_name, gate_pos, None))

        return gate_list

class H2O(ModelFromK):
    name = "H2O"

    def __init__(self, p: int, c: int, l: int, structure_list: List[int], op_pool: Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = _H2O_QUBITS
        self.num_active_electrons = 4
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('default.qubit', wires=self.num_qubits)
        self.H = qml.utils.sparse_hamiltonian(_H2O_HAM)
        self.hf = qml.qchem.hf_state(self.num_active_electrons, self.num_qubits)

    #@qml.template
    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                num_params = gate_obj.num_params
                wires = gate_dict[gate_name]
                if num_params > 0:
                    gate_params = []
                    for j in range(num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                else:
                    gate_params = None
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                qml_gate_obj.getOp()

    def constructFullCirc(self):
        @qml.qnode(self.dev,diff_method="parameter-shift")
        def fullCirc(extracted_params):
            self.backboneCirc(extracted_params)
            return qml.expval(qml.SparseHamiltonian(self.H, wires=range(self.num_qubits)))
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        return circ_func(extracted_params)

    def getLoss(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return self.costFunc(extracted_params)

    def getReward(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return -self.costFunc(extracted_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.costFunc)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]

        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_pos = gate_dict[gate_name]
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                gate_num_params = gate_obj.num_params
                if gate_num_params > 0:
                    gate_params = []
                    for j in range(gate_num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    gate_list.append((gate_name, gate_pos, gate_params))
                else:
                    gate_list.append((gate_name, gate_pos, None))

        return gate_list

class WStateFiveQubit(ModelFromK):
    name = 'WStateFiveQubit'
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 5
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('default.qubit', wires=self.num_qubits)
        self.target_state = 1/np.sqrt(5)*(np.kron(ket1, np.kron(ket0, np.kron(ket0, np.kron(ket0, ket0))))+
                                          np.kron(ket0, np.kron(ket1, np.kron(ket0, np.kron(ket0, ket0)))) +
                                          np.kron(ket0, np.kron(ket0, np.kron(ket1, np.kron(ket0, ket0)))) +
                                          np.kron(ket0, np.kron(ket0, np.kron(ket0, np.kron(ket1, ket0)))) +
                                          np.kron(ket0, np.kron(ket0, np.kron(ket0, np.kron(ket0, ket1))))
                                          )
        self.observable = qml.Hermitian(np.outer(self.target_state, self.target_state), wires = [0,1,2,3,4])

    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                num_params = gate_obj.num_params
                wires = gate_dict[gate_name]
                if num_params > 0:
                    gate_params = []
                    for j in range(num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                else:
                    gate_params = None
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                qml_gate_obj.getOp()

    def constructFullCirc(self):
        @qml.qnode(self.dev)
        def fullCirc(extracted_params):
            self.backboneCirc(extracted_params)
            return qml.expval(self.observable)
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        overlap = circ_func(extracted_params)
        return 1-overlap

    def getLoss(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        cost = self.costFunc(extracted_params)
        return cost

    def getReward(self, super_circ_params):
        return 1-self.getLoss(super_circ_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.costFunc)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]
        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_pos = gate_dict[gate_name]
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                gate_num_params = gate_obj.num_params
                if gate_num_params > 0:
                    gate_params = []
                    for j in range(gate_num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    gate_list.append((gate_name, gate_pos, gate_params))
                else:
                    gate_list.append((gate_name, gate_pos, None))

        return gate_list

class VQLSDemo(ModelFromK):
    # Modified from https://pennylane.ai/qml/demos/tutorial_vqls.html
    name = 'VQLSDemo_4Q'
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.sample_shots = 10**6
        self.n_qubits = 4
        self.tot_qubits = self.n_qubits + 1
        self.ancilla_idx = self.n_qubits
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev_mu = qml.device("lightning.qubit", wires=self.tot_qubits)
        self.div_x = qml.device("lightning.qubit", wires = self.n_qubits, shots = self.sample_shots)
        """
        J = 0.1
        zeta = 1
        eta = 0.2
        A = zeta * I + J X_1 + J X_2  + eta Z_3 Z_4
        |b> = H|0>
        """
        self.J = 0.1
        self.zeta = 1
        self.eta = 0.2
        self.coeff = np.array([self.zeta, self.J, self.J, self.eta])

    def U_b(self):
        """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
        for idx in range(self.n_qubits):
            qml.Hadamard(wires=idx)

    def CA(self, idx):
        """Controlled versions of the unitary components A_l of the problem matrix A."""
        if idx == 0:
            # identity
            None

        elif idx == 1:
            # X_0
            qml.CNOT(wires=[self.ancilla_idx, 0])

        elif idx == 2:
            # X_1
            qml.CNOT(wires=[self.ancilla_idx, 1])

        elif idx == 3:
            # Z_2 Z_3
            qml.CZ(wires=[self.ancilla_idx, 2])
            qml.CZ(wires=[self.ancilla_idx, 3])

    def backboneCirc(self, extracted_params):
        for idx in range(self.n_qubits):
            qml.Hadamard(wires=idx)
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                num_params = gate_obj.num_params
                wires = gate_dict[gate_name]
                if num_params > 0:
                    gate_params = []
                    for j in range(num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                else:
                    gate_params = None
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                qml_gate_obj.getOp()

    def constructFullCirc(self):
        @qml.qnode(self.dev_mu, diff_method='adjoint')
        def local_hadamard_test(weights, l=None, lp=None, j=None, part=None):

            # First Hadamard gate applied to the ancillary qubit.
            qml.Hadamard(wires=self.ancilla_idx)

            # For estimating the imaginary part of the coefficient "mu", we must add a "-i"
            # phase gate.
            if part == "Im" or part == "im":
                qml.PhaseShift(-np.pi / 2, wires=self.ancilla_idx)

            # Variational circuit generating a guess for the solution vector |x>
            self.backboneCirc(weights)

            # Controlled application of the unitary component A_l of the problem matrix A.
            self.CA(l)

            # Adjoint of the unitary U_b associated to the problem vector |b>.
            # In this specific example Adjoint(U_b) = U_b.
            self.U_b()

            # Controlled Z operator at position j. If j = -1, apply the identity.
            if j != -1:
                qml.CZ(wires=[self.ancilla_idx, j])

            # Unitary U_b associated to the problem vector |b>.
            self.U_b()

            # Controlled application of Adjoint(A_lp).
            # In this specific example Adjoint(A_lp) = A_lp.
            self.CA(lp)

            # Second Hadamard gate applied to the ancillary qubit.
            qml.Hadamard(wires=self.ancilla_idx)

            # Expectation value of Z for the ancillary qubit.
            return qml.expval(qml.PauliZ(wires=self.ancilla_idx))

        def mu(weights, l=None, lp=None, j=None):
            """Generates the coefficients to compute the "local" cost function C_L."""

            mu_real = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Re")
            mu_imag = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Im")

            return mu_real + 1.0j * mu_imag

        def psi_norm(weights):
            """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
            norm = 0.0

            for l in range(0, len(self.coeff)):
                for lp in range(0, len(self.coeff)):
                    norm = norm + self.coeff[l] * np.conj(self.coeff[lp]) * mu(weights, l, lp, -1)

            return abs(norm)

        def cost_loc(weights):
            """Local version of the cost function. Tends to zero when A|x> is proportional to |b>."""
            mu_sum = 0.0

            for l in range(0, len(self.coeff)):
                for lp in range(0, len(self.coeff)):
                    for j in range(0, self.n_qubits):
                        mu_sum = mu_sum + self.coeff[l] * np.conj(self.coeff[lp]) * mu(weights, l, lp, j)

            mu_sum = abs(mu_sum)

            # Cost function C_L
            return 0.5 - 0.5 * mu_sum / (self.n_qubits * psi_norm(weights))

        return cost_loc

    def costFunc(self, extracted_params):
        cost = self.constructFullCirc()
        return cost(extracted_params)

    def getLoss(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        cost = self.costFunc(extracted_params)
        return cost

    def getReward(self, super_circ_params):
        loss = self.getLoss(super_circ_params)
        # scale the reward
        return np.exp(-10*loss)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.costFunc)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]
        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_pos = gate_dict[gate_name]
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                gate_num_params = gate_obj.num_params
                if gate_num_params > 0:
                    gate_params = []
                    for j in range(gate_num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    gate_list.append((gate_name, gate_pos, gate_params))
                else:
                    gate_list.append((gate_name, gate_pos, None))

        return gate_list

    def getQuantumSolution(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)

        @qml.qnode(self.div_x)
        def prepare_and_sample(weights):
            # Variational circuit generating a guess for the solution vector |x>
            self.backboneCirc(weights)

            # We assume that the system is measured in the computational basis.
            # then sampling the device will give us a value of 0 or 1 for each qubit (n_qubits)
            # this will be repeated for the total number of shots provided (n_shots)
            return qml.sample()

        raw_samples = prepare_and_sample(extracted_params)

        # convert the raw samples (bit strings) into integers and count them
        samples = []
        for sam in raw_samples:
            samples.append(int("".join(str(bs) for bs in sam), base=2))

        q_probs = np.bincount(samples) / self.sample_shots

        return q_probs

    def getClassicalSolution(self):
        Id = np.identity(2)
        Z = np.array([[1, 0], [0, -1]])
        X = np.array([[0, 1], [1, 0]])

        A_0 = np.identity(2**self.n_qubits)
        A_1 = np.kron(X, np.kron(Id, np.kron(Id, Id)))
        A_2 = np.kron(Id, np.kron(X, np.kron(Id, Id)))

        A_3 = np.kron(Id, np.kron(Id, np.kron(Z, Z)))


        A_num = self.coeff[0] * A_0 + self.coeff[1] * A_1 + self.coeff[2] * A_2 + self.coeff[3] * A_3
        b = np.ones(2 ** self.n_qubits) / np.sqrt(2 ** self.n_qubits)

        print("A = \n", A_num)
        print("b = \n", b)

        A_inv = np.linalg.inv(A_num)
        x = np.dot(A_inv, b)

        c_probs = (x / np.linalg.norm(x)) ** 2

        return c_probs

class VQLSDemo5Q(ModelFromK):
    # Modified from https://pennylane.ai/qml/demos/tutorial_vqls.html
    name = 'VQLSDemo_5Q'
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.sample_shots = 10**6
        self.n_qubits = 5
        self.tot_qubits = self.n_qubits + 1
        self.ancilla_idx = self.n_qubits
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev_mu = qml.device("lightning.qubit", wires=self.tot_qubits)
        self.div_x = qml.device("lightning.qubit", wires = self.n_qubits, shots = self.sample_shots)
        """
        J = 0.1
        zeta = 1
        eta = 0.2
        gamma = 0.1
        A = zeta * I + J X_1 + J X_2 + eta Z_3 Z_4 + gamma Z_4 Z_5
        |b> = H|0>
        """
        self.J = 0.1
        self.zeta = 1
        self.eta = 0.2
        self.gamma = 0.1
        self.coeff = np.array([self.zeta, self.J, self.J, self.eta, self.gamma])

    def U_b(self):
        """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
        for idx in range(self.n_qubits):
            qml.Hadamard(wires=idx)

    def CA(self, idx):
        """Controlled versions of the unitary components A_l of the problem matrix A."""
        if idx == 0:
            # identity
            None

        elif idx == 1:
            # X_0
            qml.CNOT(wires=[self.ancilla_idx, 0])

        elif idx == 2:
            # X_1
            qml.CNOT(wires=[self.ancilla_idx, 1])

        elif idx == 3:
            # Z_2 Z_3
            qml.CZ(wires=[self.ancilla_idx, 2])
            qml.CZ(wires=[self.ancilla_idx, 3])

        elif idx == 4:
            # Z_3 Z_4
            qml.CZ(wires=[self.ancilla_idx, 3])
            qml.CZ(wires=[self.ancilla_idx, 4])

    def backboneCirc(self, extracted_params):
        for idx in range(self.n_qubits):
            qml.Hadamard(wires=idx)
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                num_params = gate_obj.num_params
                wires = gate_dict[gate_name]
                if num_params > 0:
                    gate_params = []
                    for j in range(num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                else:
                    gate_params = None
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                qml_gate_obj.getOp()

    def constructFullCirc(self):
        @qml.qnode(self.dev_mu, diff_method="adjoint")
        def local_hadamard_test(weights, l=None, lp=None, j=None, part=None):

            # First Hadamard gate applied to the ancillary qubit.
            qml.Hadamard(wires=self.ancilla_idx)

            # For estimating the imaginary part of the coefficient "mu", we must add a "-i"
            # phase gate.
            if part == "Im" or part == "im":
                qml.PhaseShift(-np.pi / 2, wires=self.ancilla_idx)

            # Variational circuit generating a guess for the solution vector |x>
            self.backboneCirc(weights)

            # Controlled application of the unitary component A_l of the problem matrix A.
            self.CA(l)

            # Adjoint of the unitary U_b associated to the problem vector |b>.
            # In this specific example Adjoint(U_b) = U_b.
            self.U_b()

            # Controlled Z operator at position j. If j = -1, apply the identity.
            if j != -1:
                qml.CZ(wires=[self.ancilla_idx, j])

            # Unitary U_b associated to the problem vector |b>.
            self.U_b()

            # Controlled application of Adjoint(A_lp).
            # In this specific example Adjoint(A_lp) = A_lp.
            self.CA(lp)

            # Second Hadamard gate applied to the ancillary qubit.
            qml.Hadamard(wires=self.ancilla_idx)

            # Expectation value of Z for the ancillary qubit.
            return qml.expval(qml.PauliZ(wires=self.ancilla_idx))

        def mu(weights, l=None, lp=None, j=None):
            """Generates the coefficients to compute the "local" cost function C_L."""

            mu_real = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Re")
            mu_imag = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Im")

            return mu_real + 1.0j * mu_imag

        def psi_norm(weights):
            """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
            norm = 0.0

            for l in range(0, len(self.coeff)):
                for lp in range(0, len(self.coeff)):
                    norm = norm + self.coeff[l] * np.conj(self.coeff[lp]) * mu(weights, l, lp, -1)

            return abs(norm)

        def cost_loc(weights):
            """Local version of the cost function. Tends to zero when A|x> is proportional to |b>."""
            mu_sum = 0.0

            for l in range(0, len(self.coeff)):
                for lp in range(0, len(self.coeff)):
                    for j in range(0, self.n_qubits):
                        mu_sum = mu_sum + self.coeff[l] * np.conj(self.coeff[lp]) * mu(weights, l, lp, j)

            mu_sum = abs(mu_sum)

            # Cost function C_L
            return 0.5 - 0.5 * mu_sum / (self.n_qubits * psi_norm(weights))

        return cost_loc

    def costFunc(self, extracted_params):
        cost = self.constructFullCirc()
        return cost(extracted_params)

    def getLoss(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        cost = self.costFunc(extracted_params)
        return cost

    def getReward(self, super_circ_params):
        loss = self.getLoss(super_circ_params)
        # scale the reward
        return np.exp(-30*loss)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.costFunc)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]
        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_pos = gate_dict[gate_name]
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                gate_num_params = gate_obj.num_params
                if gate_num_params > 0:
                    gate_params = []
                    for j in range(gate_num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    gate_list.append((gate_name, gate_pos, gate_params))
                else:
                    gate_list.append((gate_name, gate_pos, None))

        return gate_list

    def getQuantumSolution(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)

        @qml.qnode(self.div_x)
        def prepare_and_sample(weights):
            # Variational circuit generating a guess for the solution vector |x>
            self.backboneCirc(weights)

            # We assume that the system is measured in the computational basis.
            # then sampling the device will give us a value of 0 or 1 for each qubit (n_qubits)
            # this will be repeated for the total number of shots provided (n_shots)
            return qml.sample()

        raw_samples = prepare_and_sample(extracted_params)

        # convert the raw samples (bit strings) into integers and count them
        samples = []
        for sam in raw_samples:
            samples.append(int("".join(str(bs) for bs in sam), base=2))

        q_probs = np.bincount(samples) / self.sample_shots

        return q_probs

    def getClassicalSolution(self):
        Id = np.identity(2)
        Z = np.array([[1, 0], [0, -1]])
        X = np.array([[0, 1], [1, 0]])

        A_0 = np.identity(2**self.n_qubits)
        A_1 = np.kron(X, np.kron(Id, np.kron(Id, np.kron(Id, Id))))
        A_2 = np.kron(Id, np.kron(X, np.kron(Id, np.kron(Id, Id))))
        A_3 = np.kron(Id, np.kron(Id, np.kron(Z, np.kron(Z, Id))))
        A_4 = np.kron(Id, np.kron(Id, np.kron(Id, np.kron(Z, Z))))


        A_num = self.coeff[0] * A_0 + self.coeff[1] * A_1 + self.coeff[2] * A_2 + self.coeff[3] * A_3 + self.coeff[4] * A_4
        b = np.ones(2 ** self.n_qubits) / np.sqrt(2 ** self.n_qubits)

        print("A = \n", A_num)
        print("b = \n", b)

        A_inv = np.linalg.inv(A_num)
        x = np.dot(A_inv, b)

        c_probs = (x / np.linalg.norm(x)) ** 2

        return c_probs

class QAOAVQCDemo(ModelFromK):
    # Modified from https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut.html
    name = 'QAOAVQCDemo_7Q'
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        """
        The cost Hamiltonian:
        C_alpha = 1/2 * (1 - Z_j Z_k)
        (j, k) is an edge of the graph
        """
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.n_qubits = 7
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device("lightning.qubit", wires=self.n_qubits, shots=1)
        self.dev_train = qml.device("lightning.qubit", wires=self.n_qubits)
        self.n_samples = 100
        self.pauli_z = [[1, 0], [0, -1]]
        self.pauli_z_2 = np.kron(self.pauli_z, self.pauli_z, requires_grad=False)
        self.graph = [(0, 1), (0, 2),  (2, 3), (1, 4), (2, 4), (0 ,5),  (3, 6), (1,6)]

    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                num_params = gate_obj.num_params
                wires = gate_dict[gate_name]
                if num_params > 0:
                    gate_params = []
                    for j in range(num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                else:
                    gate_params = None
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                qml_gate_obj.getOp()

    def bitstring_to_int(self, bit_string_sample):
        bit_string = "".join(str(bs) for bs in bit_string_sample)
        return int(bit_string, base=2)

    def sample_result_to_str(self, bit_string_sample):
        return "".join(str(bs) for bs in bit_string_sample)

    def objective(self, extracted_params):
        @qml.qnode(self.dev_train)
        def circuit(weights, edge=None):
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)

            self.backboneCirc(weights)

            if edge is None:
                return qml.sample()

            return qml.expval(qml.Hermitian(self.pauli_z_2, wires=edge))

        neg_obj = 0
        for edge in self.graph:
            neg_obj -= 0.5*(1-circuit(extracted_params, edge=edge))
        return neg_obj

    def getLoss(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        loss = self.objective(extracted_params)
        return loss

    def getReward(self, super_circ_params):
        loss = self.getLoss(super_circ_params)
        return -loss

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.objective)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]
        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_pos = gate_dict[gate_name]
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                gate_num_params = gate_obj.num_params
                if gate_num_params > 0:
                    gate_params = []
                    for j in range(gate_num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    gate_list.append((gate_name, gate_pos, gate_params))
                else:
                    gate_list.append((gate_name, gate_pos, None))

        return gate_list

    def getQuantumSolution(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)

        @qml.qnode(self.dev)
        def circuit(weights, edge=None):
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)

            self.backboneCirc(weights)

            if edge is None:
                return qml.sample()

            return qml.expval(qml.Hermitian(self.pauli_z_2, wires=edge))

        bit_strings = []
        original_samples = []
        for i in range(0, self.n_samples):
            bits = circuit(extracted_params, edge=None)
            bit_strings.append(self.bitstring_to_int(bits))
            original_samples.append(self.sample_result_to_str(bits))

        counts = np.bincount(np.array(bit_strings))
        most_freq_bit_string = np.argmax(counts)
        original_samples = Counter(original_samples)
        original_samples = dict(OrderedDict(original_samples.most_common()))
        return "{:07b}".format(most_freq_bit_string), original_samples

    def getClassicalSolution(self):
        """
        More than one solutions:
        '[1, 0, 0, 1, 1, 0, 0]': 7.0,
        '[0, 1, 1, 0, 0, 1, 0]': 7.0,
        '[0, 1, 1, 1, 0, 1, 0]': 7.0,
        '[1, 0, 0, 0, 1, 0, 1]': 7.0,
        '[1, 0, 0, 1, 1, 0, 1]': 7.0,
        '[0, 1, 1, 0, 0, 1, 1]': 7.0
        :return:
        """
        return {'[1, 0, 0, 1, 1, 0, 0]': 7.0, '[0, 1, 1, 0, 0, 1, 0]': 7.0, '[0, 1, 1, 1, 0, 1, 0]': 7.0, '[1, 0, 0, 0, 1, 0, 1]': 7.0, '[1, 0, 0, 1, 1, 0, 1]': 7.0, '[0, 1, 1, 0, 0, 1, 1]': 7.0}

class QAOAWeightedVQCDemo(ModelFromK):
    # Modified from https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut.html
    name = 'QAOAWeightedVQCDemo_5Q'
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        """
        The cost Hamiltonian:
        C_alpha = 1/2 * (1 - Z_j Z_k)*w
        (j, k) is an edge of the graph
        optimal function value: 3.3
        optimal value: [0. 1. 1. 1. 0. 1. 0.]
        """
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.n_qubits = 5
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device("lightning.qubit", wires=self.n_qubits, shots=1)
        self.dev_train = qml.device("lightning.qubit", wires=self.n_qubits)
        self.n_samples = 100
        self.pauli_z = [[1, 0], [0, -1]]
        self.pauli_z_2 = np.kron(self.pauli_z, self.pauli_z, requires_grad=False)
        self.graph = [(0, 1, 1), (0, 2, 2),  (2, 3, 3), (1, 4, 4), (2, 4, 5), (0, 4, 6)]
    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                num_params = gate_obj.num_params
                wires = gate_dict[gate_name]
                if num_params > 0:
                    gate_params = []
                    for j in range(num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                else:
                    gate_params = None
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                qml_gate_obj.getOp()

    def bitstring_to_int(self, bit_string_sample):
        bit_string = "".join(str(bs) for bs in bit_string_sample)
        return int(bit_string, base=2)

    def sample_result_to_str(self, bit_string_sample):
        return "".join(str(bs) for bs in bit_string_sample)

    def objective(self, extracted_params):
        @qml.qnode(self.dev_train)
        def circuit(weights, edge=None):
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)

            self.backboneCirc(weights)

            if edge is None:
                return qml.sample()
            wires = [edge[0], edge[1]]

            return qml.expval(qml.Hermitian(self.pauli_z_2, wires=wires))

        neg_obj = 0
        for edge in self.graph:
            weight = edge[2]
            neg_obj -= 0.5*(1-circuit(extracted_params, edge=edge))*weight
        return neg_obj

    def getLoss(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        loss = self.objective(extracted_params)
        return loss

    def getReward(self, super_circ_params):
        loss = self.getLoss(super_circ_params)
        return -loss

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.objective)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]
        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_pos = gate_dict[gate_name]
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                gate_num_params = gate_obj.num_params
                if gate_num_params > 0:
                    gate_params = []
                    for j in range(gate_num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    gate_list.append((gate_name, gate_pos, gate_params))
                else:
                    gate_list.append((gate_name, gate_pos, None))

        return gate_list

    def getQuantumSolution(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)

        @qml.qnode(self.dev)
        def circuit(weights, edge=None):
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)

            self.backboneCirc(weights)

            if edge is None:
                return qml.sample()

            wires = [edge[0], edge[1]]

            return qml.expval(qml.Hermitian(self.pauli_z_2, wires=wires))

        bit_strings = []
        original_samples = []
        for i in range(0, self.n_samples):
            bits = circuit(extracted_params, edge=None)
            bit_strings.append(self.bitstring_to_int(bits))
            original_samples.append(self.sample_result_to_str(bits))

        counts = np.bincount(np.array(bit_strings))
        most_freq_bit_string = np.argmax(counts)
        original_samples = Counter(original_samples)
        original_samples = dict(OrderedDict(original_samples.most_common()))
        return "{:05b}".format(most_freq_bit_string), original_samples

    def getClassicalSolution(self):
        return {'[1, 1, 1, 0, 0]': 18.0, '[0, 0, 0, 1, 1]': 18.0}
