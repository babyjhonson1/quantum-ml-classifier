from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, SparsePauliOp
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

OBS_OPERATOR = SparsePauliOp('IZ')

def build_circuit(x, theta, n_layers):
    qc = QuantumCircuit(2)

    qc.ry(x[0], 0)
    qc.ry(x[1], 1)

    param_index = 0
    for i in range(n_layers):
        qc.ry(theta[param_index], 0)
        param_index += 1
        qc.ry(theta[param_index], 1)
        param_index += 1
        if i % 2 == 0:
            qc.cx(0, 1)
        else:
            qc.cx(1, 0)

    return qc

def expectation(x, theta, n_layers, OBS_OPERATOR):
    qc = build_circuit(x, theta, n_layers)
    state = Statevector.from_instruction(qc)
    return state.expectation_value(OBS_OPERATOR).real

def predict_prob(X, theta, n_layers):
    probabilities = np.zeros(len(X))

    for i in range(len(X)):
        exp = expectation(X[i], theta, n_layers, OBS_OPERATOR)
        probabilities[i] = (1 - exp) / 2
    
    return probabilities

def binary_cross_entropy(y, prob):
    eps = 1e-9
    prob = np.clip(prob, eps, 1 - eps)
    loss = -sum(y * np.log(prob) + (1 - y) * np.log(1 - prob)) / len(y)
    return loss

def loss(X, y, theta, n_layers):
    prob = predict_prob(X, theta, n_layers)
    loss_val = binary_cross_entropy(y, prob)
    return loss_val

def predict_labels(X, theta, n_layers):
    prob = predict_prob(X, theta, n_layers)
    return (prob >= 0.5).astype(int)


data = np.load('data/moons.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

rng = np.random.default_rng(42)

def objective(theta):
    loss_val = loss(X_train, y_train, theta, n_layers)
    loss_hist.append(loss_val)
    # if len(loss_hist) % 20 == 0:
    #     print(f"iter: {len(loss_hist)}, loss = {loss_val: .5f}")
    return loss_val

for n_layers in range(2, 10):
    init_theta = rng.uniform(-2*np.pi, 2*np.pi, 2 * n_layers)

    loss_hist = []
    print(f"=======TRAINING WITH n_layers = {n_layers}=======")
    result = minimize(
        fun=objective,
        x0=init_theta,
        method='COBYLA',
        options={'maxiter': 200}
    )

    best_theta = result.x
    y_train_pred = predict_labels(X_train, best_theta, n_layers)
    y_test_pred = predict_labels(X_test, best_theta, n_layers)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("Optimization success:", result.success)
    print("Final loss:", result.fun)
    print("Best theta:", best_theta)
    print(f"Quantum train accuracy: {train_acc:.4f}")
    print(f"Quantum test accuracy: {test_acc:.4f}")

    plt.figure()
    plt.plot(loss_hist)
    plt.xlabel("Optimization step")
    plt.ylabel("Binary cross-entropy")
    plt.title("Quantum classifier training loss")
    plt.tight_layout()
    plt.savefig(f"figures/loss_{n_layers}_n_layers.png", dpi=200)