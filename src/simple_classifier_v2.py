from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

OBS_OPERATOR = SparsePauliOp('IZ')

def build_circuit(x, theta, n_layers):
    qc = QuantumCircuit(2)

    param_index = 0
    for i in range(n_layers):
        qc.ry(x[0], 0)
        qc.ry(x[1], 1)

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

def plot_decision_boundary(X, y, theta, n_layers, output_path):
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    probs = predict_prob(grid, theta, n_layers)
    probs = probs.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, probs, levels=30, alpha=0.7)
    plt.contour(xx, yy, probs, levels=[0.5], linewidths=2)

    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=35)

    plt.title(f"Decision boundary, layers={n_layers}")
    plt.colorbar(label="p(class=1)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


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

best_theta = []
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

    cur_theta = result.x
    best_theta.append(cur_theta)
    y_train_pred = predict_labels(X_train, cur_theta, n_layers)
    y_test_pred = predict_labels(X_test, cur_theta, n_layers)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("Final loss:", result.fun)
    print(f"Quantum train accuracy: {train_acc:.4f}")
    print(f"Quantum test accuracy: {test_acc:.4f}")

    plt.figure()
    plt.plot(loss_hist)
    plt.xlabel("Optimization step")
    plt.ylabel("Binary cross-entropy")
    plt.title("Quantum classifier training loss")
    plt.tight_layout()
    plt.savefig(f"figures/v2/loss_{n_layers}_n_layers.png", dpi=200)

    plot_decision_boundary(X_test, y_test, best_theta[n_layers - 2], n_layers, f'figures/v2/decision_boundary_{n_layers}.png')