from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
scaler = MinMaxScaler((0, 2*np.pi))
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

np.savez(
    'data/moons.npz',
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    X_scaled=X_scaled,
    y=y
)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y)
plt.savefig("figures/moons.png")

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"LogReg acc: {acc}")

model = SVC(kernel='rbf', gamma='scale', C=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"SVM-RBF acc: {accuracy_score(y_test, y_pred)}")
