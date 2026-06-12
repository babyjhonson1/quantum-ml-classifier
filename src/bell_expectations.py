from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, SparsePauliOp

simulator = AerSimulator()

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)

state = Statevector.from_instruction(qc)
print(state)

op_zz = SparsePauliOp('ZZ')
op_xx = SparsePauliOp('XX')
op_yy = SparsePauliOp('YY')
op_iz = SparsePauliOp('IZ')
op_zi = SparsePauliOp('ZI')

print(state.expectation_value(op_zz))
print(state.expectation_value(op_xx))
print(state.expectation_value(op_yy))
print(state.expectation_value(op_iz))
print(state.expectation_value(op_zi))
# qc.measure([0, 1], [0, 1])
# comp_sim = transpile(qc, simulator)
# job = simulator.run(comp_sim, shots=1000)
# result = job.result()
# counts = result.get_counts()
# print(qc)
# print(counts)