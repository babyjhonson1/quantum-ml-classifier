from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def count_and_qc(qc, simulator, shots=1000):
    measured_qc = qc.copy()
    measured_qc.measure([0, 1], [0, 1])

    compiled_sim = transpile(measured_qc, simulator)
    job = simulator.run(compiled_sim, shots=shots)
    result = job.result()
    counts = result.get_counts()

    print(measured_qc)
    print(counts)

simulator = AerSimulator()

qc = QuantumCircuit(2, 2)
# qc.initialize([0., 0.5**0.5, -0.5**0.5, 0.], [0, 1]) #bell entangled state psi^-
# count_and_qc(qc, simulator)

qc.h(0)
qc.cx(0, 1)
count_and_qc(qc, simulator)