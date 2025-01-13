from qas.qml_models.qml_gate_ops import QMLPool, SUPPORTED_OPS_DICT
from qas.mcts.mcts_state import StateOfMCTS
from copy import deepcopy
from typing import (
    List,
    Sequence,
    Callable,
    Optional,
    Union,
    Set
)
class QMLStateBasicGates(StateOfMCTS):
    name = 'QMLStateBasicGates'

    def __init__(self, current_k:List[int]=[], op_pool:QMLPool=None, maxDepth = 30, qubit_with_actions:Set = None, gate_limit_dict:Optional[dict] = None):
        self.max_depth = maxDepth
        self.current_k = current_k
        self.current_depth = len(self.current_k)
        assert self.current_depth <= self.max_depth
        self.pool_obj = op_pool
        self.op_name_dict = op_pool.pool
        self.pool_keys = list(op_pool.pool.keys())
        self.single_qubit_gate_names = op_pool.single_qubit_gates
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

    def getLegalActions(self):
        if self.current_depth == self.max_depth:
            return []
        actions = []
        for key in self.pool_keys:
            if self.verifyDesirableAction(key):
                actions.append(key)
        return actions

    def stackOpsOnQubit(self, k):
        stacked_ops = [[] for _ in range(self.pool_obj.num_qubits)]
        for action in k:
            op = self.op_name_dict[action]
            assert len(op.keys()) == 1 # in case some weird things happen
            op_name = list(op.keys())[0]
            op_qubit = op[op_name]
            for qubit in op_qubit:
                if op_name != "PlaceHolder":
                    stacked_ops[qubit].append(action)
        return stacked_ops

    def verifyDesirableAction(self, action:int):
        # many change with different tasks
        op = self.op_name_dict[action]
        assert len(op.keys()) == 1 # in case some weird things happen
        op_name = list(op.keys())[0]
        op_qubit = op[op_name]
        op_obj = SUPPORTED_OPS_DICT[op_name]
        op_num_params = op_obj.num_params
        stacked_ops = self.stackOpsOnQubit(self.current_k)

        # don't want two consecutive parameterized gates or Pauli gates or CNOT gates
        if len(self.current_k)>=1:
            if len(op_qubit) == 1:
                if op_num_params>0:
                    if len(stacked_ops[op_qubit[0]])>=1:
                        if stacked_ops[op_qubit[0]][-1] == action:
                            return False
                elif op_name in ['PauliX', 'PauliY', 'PauliZ','Hadamard']:
                    if len(stacked_ops[op_qubit[0]])>=1:
                        if stacked_ops[op_qubit[0]][-1] == action:
                            return False
                elif op_name == 'T':
                    if len(stacked_ops[op_qubit[0]]) >= 1:
                        last_action = stacked_ops[op_qubit[0]][-1]
                        last_op = self.op_name_dict[last_action]
                        last_op_name = list(last_op.keys())[0]
                        if last_op_name == 'Tdg':
                            return False
                        if action == last_action and 'S' in self.single_qubit_gate_names:
                            return False
                elif op_name == 'Tdg':
                    if len(stacked_ops[op_qubit[0]]) >= 1:
                        last_action = stacked_ops[op_qubit[0]][-1]
                        last_op = self.op_name_dict[last_action]
                        last_op_name = list(last_op.keys())[0]
                        if last_op_name == 'T':
                            return False
                        if action == last_action and 'Sdg' in self.single_qubit_gate_names:
                            return False
                elif op_name == 'S':
                    if len(stacked_ops[op_qubit[0]]) >= 1:
                        last_action = stacked_ops[op_qubit[0]][-1]
                        last_op = self.op_name_dict[last_action]
                        last_op_name = list(last_op.keys())[0]
                        if last_op_name == 'Sdg':
                            return False
                        if action == last_action and 'PauliZ' in self.single_qubit_gate_names:
                            return False
                elif op_name == 'Sdg':
                    if len(stacked_ops[op_qubit[0]]) >= 1:
                        last_action = stacked_ops[op_qubit[0]][-1]
                        last_op = self.op_name_dict[last_action]
                        last_op_name = list(last_op.keys())[0]
                        if last_op_name == 'S':
                            return False
                        if action == last_action and 'PauliZ' in self.single_qubit_gate_names:
                            return False


            if len(op_qubit) == 2:
                if len(stacked_ops[op_qubit[0]])>=1 and len(stacked_ops[op_qubit[1]])>=1:
                    if stacked_ops[op_qubit[0]][-1] == action and stacked_ops[op_qubit[1]][-1] == action:
                        if op_num_params>0: return False
                        if op_name in ['CNOT', 'CZ', 'CY']: return False

        if len(op_qubit) == 2 and op_qubit[0] not in self.qubit_with_actions:
            return False

        if op_name == "PlaceHolder" and op_qubit[0] not in self.qubit_with_actions:
            return False

        # control the number of gates
        if op_name in set(self.gate_limit_dict.keys()):
            if self.gate_count[op_name] >= self.gate_limit_dict[op_name]:
                return False

        return True

    def takeAction(self, action:int):
        new_qubit_with_actions = deepcopy(self.qubit_with_actions)
        op = self.op_name_dict[action]
        assert len(op.keys()) == 1  # in case some weird things happen
        op_name = list(op.keys())[0]
        op_qubit = op[op_name]
        if op_name != "PlaceHolder":
            new_qubit_with_actions.add(op_qubit[-1])
        new_path = self.current_k + [action]
        new_state = QMLStateBasicGates(current_k=new_path, op_pool=self.pool_obj, maxDepth=self.max_depth,
                                       qubit_with_actions=new_qubit_with_actions, gate_limit_dict=self.gate_limit_dict)
        new_state.state = action
        return new_state

    def isTerminal(self):
        return self.current_depth>=self.max_depth

    def getReward(self):
        return

    def getCurrK(self):
        return self.current_k

    def __repr__(self):
        disp = ""
        for i, p in enumerate(self.current_k):
            disp = disp + "OpAtDepth: {}\tOpKey: {}\tOpName: {}\n".format(i, p, self.op_name_dict[p])
        return disp

class QMLStateBasicGatesNoRestrictions(StateOfMCTS):
    name = 'QMLStateBasicGatesNoRestrictions'

    def __init__(self, current_k:List[int]=[], op_pool:QMLPool=None, maxDepth = 30, qubit_with_actions:Set = None, gate_limit_dict:Optional[dict] = None):
        self.max_depth = maxDepth
        self.current_k = current_k
        self.current_depth = len(self.current_k)
        assert self.current_depth <= self.max_depth
        self.pool_obj = op_pool
        self.op_name_dict = op_pool.pool
        self.pool_keys = list(op_pool.pool.keys())
        self.single_qubit_gate_names = op_pool.single_qubit_gates
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

    def getLegalActions(self):
        if self.current_depth == self.max_depth:
            return []
        actions = []
        for key in self.pool_keys:
            actions.append(key)
        return actions


    def takeAction(self, action:int):
        new_qubit_with_actions = deepcopy(self.qubit_with_actions)
        op = self.op_name_dict[action]
        assert len(op.keys()) == 1  # in case some weird things happen
        op_name = list(op.keys())[0]
        op_qubit = op[op_name]
        if op_name != "PlaceHolder":
            new_qubit_with_actions.add(op_qubit[-1])
        new_path = self.current_k + [action]
        new_state = QMLStateBasicGates(current_k=new_path, op_pool=self.pool_obj, maxDepth=self.max_depth,
                                       qubit_with_actions=new_qubit_with_actions, gate_limit_dict=self.gate_limit_dict)
        new_state.state = action
        return new_state

    def isTerminal(self):
        return self.current_depth>=self.max_depth

    def getReward(self):
        return 0.

    def getCurrK(self):
        return self.current_k

    def __repr__(self):
        disp = ""
        for i, p in enumerate(self.current_k):
            disp = disp + "OpAtDepth: {}\tOpKey: {}\tOpName: {}\n".format(i, p, self.op_name_dict[p])
        return disp
