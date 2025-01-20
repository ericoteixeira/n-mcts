import os
os.environ["OMP_NUM_THREADS"] = "12"

from qas.mcts.mcts import search, TreeNode, circuitModelTuning
from qas.qml_models.qml_gate_ops import QMLPool
from qas.qml_models.kagome_heisenberg_model import KagomeHeisenberg
from qas.mcts.qml_mcts_state import QMLStateBasicGates

import json
import numpy as np
import pennylane as qml
import time


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

    model = KagomeHeisenberg
    state_class = QMLStateBasicGates

    marker = nowtime()
    task = model.name + "_" + state_class.name
    filename = task + "_"+ marker + '.json'
    print(task)
    init_qubit_with_actions = set()
    two_qubit_gate = ["CNOT"]
    single_qubit_gate = ["Rot", "PlaceHolder"]
    # connection graph of ibmq_guadalupe
    connection_graph = [[0,1],[1,4],[1,2],[1,0],[2,1],[2,3],[3,2],[3,5],[4,1],[4,7],[5,8],[5,3],[6,7],[7,4],[7,10],[7,6],[8,5],[8,9],[8,11],[9,8],[10,12],[10,7],[11,8],[11,14],[12,15],[12,10],[12,13],[13,14],[13,12],[14,13],[14,11],[15,12]]
    pool = QMLPool(16, single_qubit_gate, two_qubit_gate, complete_undirected_graph=False,  two_qubit_gate_map=connection_graph)
    print(pool)
    p = 500
    l = 3
    c = len(pool)
    ph_count_limit = p
    gate_limit = {"CNOT": p}

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


    init_params = np.random.randn(p, c, l)

    final_params, final_best_arc, final_best_node, final_best_reward, final_controller, reward_list = search(
        model=model,
        op_pool=pool,
        target_circuit_depth=p,
        init_qubit_with_controls=init_qubit_with_actions,
        init_params=init_params,
        num_iterations=200,
        num_warmup_iterations=50,
        super_circ_train_optimizer=qml.AdamOptimizer,
        super_circ_train_gradient_noise_factor=0.0,
        early_stop_threshold=18,
        early_stop_lookback_count=5,
        super_circ_train_lr=1,
        penalty_function=penalty_func,
        gate_limit_dict=gate_limit,
        warmup_arc_batchsize=500,
        search_arc_batchsize=100,
        alpha_max=2,
        alpha_decay_rate=0.99,
        prune_constant_max=0.99,
        prune_constant_min=0.80,
        max_visits_prune_threshold=5,
        min_num_children=5,
        sampling_execute_rounds=20,
        exploit_execute_rounds=5,
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
        num_epochs=500,
        k=final_best_arc,
        op_pool=pool,
        opt_callable=qml.AdamOptimizer,
        lr=0.1,
        grad_noise_factor=0,
        verbose=1,
        early_stop_threshold=-18
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
