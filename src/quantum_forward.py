from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, SparsePauliOp
import numpy as np

def build_circuit(x, theta):
    qc = QuantumCircuit(2, 2)

    #feature map
    qc.ry(x[0], 0)
    qc.ry(x[1], 1)

    #ansatz
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)

    return qc

data = np.load('data/moons.npz')
x = data['X_scaled'][0]
y = data['y'][0]
theta = [0.2, -0.3]

qc = build_circuit(x, theta)
state = Statevector.from_instruction(qc)

op_iz = SparsePauliOp('IZ')
expect_iz = state.expectation_value(op_iz).real
p_class1 = (1 + expect_iz)/2
print(qc)
print("x:", x)
print("statevector:", state)
print("<Z0>:", expect_iz)
print("p(class=1):", p_class1)
print("true label:", y)