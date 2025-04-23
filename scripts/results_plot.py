import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from pprint import pprint
cwd = os.getcwd()
print(cwd)

#----------------------------------------------------
#Modify
#----------------------------------------------------
results = '20250217-153929_FourActiveSpacePCA_QMLStateBasicGates.json'

E_FCI = -559.702929
num_qubits = 8
#----------------------------------------------------

with open(os.path.join(cwd, results)) as f:
    results = json.load(f)

print(results.keys())
pprint(results['op_list'])
print(len(results['op_list']))

# count the number of parameters
results_num_parameters = 0
for c in results['op_list']:
    if c[2] != None:
        results_num_parameters = results_num_parameters + len(c[2])
print()
print(results_num_parameters)

#circuit
with qml.tape.QuantumTape() as tape:
    for op in results['op_list']:
        gate, wires, params = op
        if gate == "Rot":
            qml.Rot(*params, wires=wires) 
        elif gate == "Hadamard":
            qml.Hadamard(wires=wires)
        elif gate == "RX":
            qml.RX(params, wires=wires)
        elif gate == "RY":
            qml.RY(params, wires=wires)
        elif gate == "RZ":
            qml.RZ(params, wires=wires)
        elif gate == "CNOT":
            qml.CNOT(wires=wires)
print(tape.draw(wire_order=range(num_qubits)))


dev = qml.device("default.qubit", wires=num_qubits)
@qml.qnode(dev)
def circuit():
    for op in results['op_list']:
        gate, wires, params = op
        if gate == "Rot":
            qml.Rot(*params, wires=wires) 
        elif gate == "Hadamard":
            qml.Hadamard(wires=wires)
        elif gate == "RX":
            qml.RX(params, wires=wires)
        elif gate == "RY":
            qml.RY(params, wires=wires)
        elif gate == "RZ":
            qml.RZ(params, wires=wires)
        elif gate == "CNOT":
            qml.CNOT(wires=wires)
    return qml.state()


#--------------------------------------------------------------------------------
#Graphs
#--------------------------------------------------------------------------------
print(qml.draw(circuit))
qml.drawer.use_style('black_white')
fig, ax = qml.draw_mpl(circuit)()
plt.savefig('fig_qaoa_circ_1.pdf')
plt.close()
results_search_rewards = [s[2] for s in results['search_reward_list']]
results_fine_tune_loss = results['fine_tune_loss']


final_energy = results_fine_tune_loss[-1]

# Plot Search Rewards
plt.figure(figsize=(8, 5))
plt.plot(list(range(len(results_search_rewards))), results_search_rewards, marker='x', label='Search Rewards')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Reward (-Energy, Ha)', fontsize=15)
#plt.title("Search Rewards", fontsize=16)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("search_rewards.pdf")
plt.show()


# Plot Fine Tune Loss
plt.figure(figsize=(8, 5))
plt.plot(list(range(len(results_fine_tune_loss))), results_fine_tune_loss, marker='o', label='Fine Tune Loss. Energy = ' + f"{final_energy:8f} Ha")
plt.axhline(y=E_FCI, color='r', linestyle='--', label=r"$E_{\text{FCI}} = $" + f"{E_FCI:.6f} Ha")
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Fine Tune Loss', fontsize=15)
#plt.title("Fine Tune Loss", fontsize=16)
plt.legend(fontsize=15)

plt.tight_layout()
plt.savefig("fine_tune_loss.pdf")
plt.show()

