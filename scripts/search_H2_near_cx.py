import os
os.environ["OMP_NUM_THREADS"] = "12"

from qas.mcts.mcts import search, TreeNode, circuitModelTuning
from qas.qml_models.qml_gate_ops import QMLPool
from qas.qml_models.qml_models_legacy import FourQubitH2_VaccumInitial
import json
import numpy as np
import pennylane as qml
import time
from qas.mcts.qml_mcts_state import QMLStateBasicGates
import pennylane.numpy as pnp


#warnings.filterwarnings("ignore")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def nowtime():
    return str(time.strftime("%Y%m%d-%H%M%S", time.localtime()))


if __name__ == "__main__":

    import shutup
    shutup.please()

    model = FourQubitH2_VaccumInitial
    # classical results:
    # converged SCF energy = -1.1145697422375
    # E(FCI) = -1.131902147424
    # converged SCF energy = -1.1145697422375  <S^2> = 0  2S+1 = 1
    # E(UHF-FCI) = -1.131902147424
    # E(FCI) = -1.131902147424
    # adaptive circuit results:
    # Total number of gates: 28
    # Total number of two-qubit control gates: 14

    state_class = QMLStateBasicGates


    marker = nowtime()
    task = model.name + "_" + state_class.name
    filename = marker + "_" + task + '.json'
    print(task)
    init_qubit_with_actions = set()
    two_qubit_gate = ["CNOT"]
    single_qubit_gate = ["Rot","PlaceHolder"]
    connection_graph = [[0,1],[1,0],[1,2],[2,1],[2,3],[3,2]]

    # set a hard limit on the number of certain gate instead of using a penalty function

    pool = QMLPool(4, single_qubit_gate, two_qubit_gate, complete_undirected_graph=False, two_qubit_gate_map=connection_graph)
    print(pool)
    p = 30
    l = 3
    c = len(pool)
    ph_count_limit = p
    gate_limit = {"CNOT": p//2}


    # penalty function:
    def penalty_func(r: float, node: TreeNode):
        k = node.state.getCurrK()
        ph_count = 0
        for op_index in k:
            op_name = list(pool[op_index].keys())[0]
            if op_name == 'PlaceHolder':
                ph_count = ph_count + 1
        if ph_count >= ph_count_limit:
            return r - (ph_count - ph_count_limit) / 1
        return r


    #init_params = np.random.randn(p, c, l)
    init_params = pnp.array(pnp.random.randn(p, c, l), requires_grad=True)

    final_params, final_best_arc, final_best_node, final_best_reward, final_controller, reward_list = search(
        model=model,
        op_pool=pool,
        target_circuit_depth=p,
        init_qubit_with_controls=init_qubit_with_actions,
        init_params=init_params,
        num_iterations=50,
        num_warmup_iterations=5,
        super_circ_train_optimizer=qml.AdamOptimizer,
        super_circ_train_gradient_noise_factor=0.0,
        early_stop_threshold=999,
        early_stop_lookback_count=1,
        super_circ_train_lr=1,
        penalty_function=penalty_func,
        gate_limit_dict=gate_limit,
        warmup_arc_batchsize=500,
        search_arc_batchsize=100,
        alpha_max=2,
        alpha_decay_rate=0.99,
        prune_constant_max=0.9,
        prune_constant_min=0.80,
        max_visits_prune_threshold=10,
        min_num_children=5,
        sampling_execute_rounds=10,
        exploit_execute_rounds=20,
        cmab_sample_policy='local_optimal',
        cmab_exploit_policy='local_optimal',
        uct_sample_policy='local_optimal',
        verbose=1,
        state_class=state_class,
        search_reset=True
    )

    final_params, loss_list = circuitModelTuning(
        initial_params=init_params,
        model=model,
        num_epochs=100,
        k=final_best_arc,
        op_pool=pool,
        opt_callable=qml.AdamOptimizer,
        lr=0.1,
        grad_noise_factor=0,
        verbose=1,
        early_stop_threshold=-999
    )

    res_dict = {
        'task': task,
        'pool': pool.pool,
        'params': final_params,
        'k': final_best_arc,
        'op_list': model(p, c, l, final_best_arc, pool).toList(final_params),
        'search_reward_list': reward_list,
        'fine_tune_loss': loss_list
    }

    with open(filename, 'w') as f:
        json.dump(res_dict, f, indent=4, cls=NpEncoder)
