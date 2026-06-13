Quantum ML Classifier на Qiskit

Этот проект реализует и исследует небольшой квантовый классификатор для задачи нелинейной бинарной классификации с использованием Qiskit.

Цель проекта — разобраться и показать полный пайплайн гибридного quantum-classical ML:

* кодирование классических данных в квантовую схему;
* параметризованный квантовый ansatz;
* считывание результата, соответствующего матрице Паули (\sigma_z);
* классическая оптимизация функции потерь;
* сравнение с классическими ML-моделями.

В проекте используется двумерный датасет make_moons. Сравниваются:

* Logistic Regression;
* RBF-SVM;
* простой квантовый классификатор;
* квантовый классификатор с data re-uploading.

⸻

Мотивация

Квантовый классификатор можно рассматривать как параметризованную функцию:

[
f_\theta(x)=
\langle 0|U^\dagger(x,\theta)OU(x,\theta)|0\rangle,
]

где:

* (x) — классические входные признаки;
* (\theta) — обучаемые параметры квантовой схемы;
* (U(x,\theta)) — параметризованная квантовая схема;
* (O) — измеряемый оператор, в этом проекте Pauli-Z observable;
* (f_\theta(x)) затем переводится в вероятность класса.

Обучение происходит гибридно: квантовая схема задаёт expectation value, а классический оптимизатор обновляет параметры (\theta).

Основная исследовательская идея проекта — проверить, как архитектура квантовой схемы влияет на качество классификации, особенно сравнить простое angle encoding и data re-uploading.

⸻

Датасет

Данные генерируются через sklearn.datasets.make_moons:

X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

Признаки масштабируются в диапазон ([0, 2\pi]), потому что дальше используются как углы вращения в квантовых гейтах:

scaler = MinMaxScaler((0, 2*np.pi))
X_scaled = scaler.fit_transform(X)

Данные делятся на train и test:

test_size=0.3
random_state=42
stratify=y

После запуска подготовки данных появляются файлы:

data/moons.npz
figures/moons.png

⸻

Классические baseline-модели

Для сравнения были обучены две классические модели.

Модель	Test accuracy
Logistic Regression	0.90
RBF-SVM	0.95

RBF-SVM используется как сильный нелинейный классический baseline.

⸻

Квантовая проверка: Bell state

Перед обучением классификатора в проекте проверяется базовый квантовый workflow.

Готовится состояние Белла:

[
|\Phi^+\rangle =
\frac{|00\rangle + |11\rangle}{\sqrt{2}}
]

и проверяются корреляции:

[
\langle ZZ\rangle = 1,
\quad
\langle XX\rangle = 1,
\quad
\langle YY\rangle = -1,
\quad
\langle IZ\rangle = 0,
\quad
\langle ZI\rangle = 0.
]

Реализация:

src/bell_expectations.py

Смысл этого шага — проверить цепочку:

QuantumCircuit → Statevector → SparsePauliOp → expectation value

⸻

Квантовая модель

Классификатор использует 2 кубита.

Выход модели строится через expectation value оператора:

[
Z_0.
]

В Qiskit он задаётся так:

SparsePauliOp("IZ")

Expectation value переводится в вероятность класса 1:

[
p(y=1|x)=\frac{1-\langle Z_0\rangle}{2}.
]

Для обучения используется binary cross-entropy:

[
\mathcal{L}(\theta)=
-\frac{1}{N}
\sum_i
\left[
y_i\log p_i + (1-y_i)\log(1-p_i)
\right].
]

Параметры оптимизируются через scipy.optimize.minimize с методом COBYLA.

⸻

QC v1: простое angle encoding

Первая версия квантового классификатора кодирует данные один раз в начале схемы:

[
|00\rangle
\xrightarrow{R_y(x_1),R_y(x_2)}
|\phi(x)\rangle
\xrightarrow{W(\theta)}
|\psi(x,\theta)\rangle.
]

Структура схемы:

RY(x0) на q0
RY(x1) на q1
для каждого слоя:
    RY(theta_k) на q0
    RY(theta_k+1) на q1
    CX

Реализация:

src/simple_classifier_v1.py

Лучший наблюдаемый результат:

Модель	Best test accuracy
Simple QC	0.8333

Простая QC-модель обучается и строит нелинейную границу, но уступает классическим baseline.

⸻

QC v2: data re-uploading

Вторая версия использует data re-uploading.

Идея: данные кодируются не один раз, а повторно на каждом слое схемы:

[
U(x,\theta)=
\prod_{\ell=1}^{L}
W_\ell(\theta_\ell)\Phi(x).
]

Структура схемы:

для каждого слоя:
    RY(x0) на q0
    RY(x1) на q1
    RY(theta_k) на q0
    RY(theta_k+1) на q1
    CX

Реализация:

src/simple_classifier_v2.py

Data re-uploading заметно улучшил качество модели.

n_layers	Train accuracy	Test accuracy	Final loss
2	0.8000	0.7333	0.4653
3	0.7643	0.8500	0.4913
4	0.8714	0.8667	0.3155
5	0.9143	0.9167	0.2600
6	0.9071	0.9333	0.2617
7	0.8214	0.7667	0.3501
8	0.9000	0.8167	0.3528
9	0.8857	0.7667	0.3117

Лучший результат:

Модель	Test accuracy
Logistic Regression	0.90
RBF-SVM	0.95
Simple QC	0.8333
Data re-uploading QC	0.9333

QC с data re-uploading приблизился к RBF-SVM и превзошёл Logistic Regression на этом датасете.

⸻

Главный результат

Главный экспериментальный результат проекта:

data re-uploading значительно увеличивает выразительность маленького двухкубитного вариационного квантового классификатора.

Простая QC-модель дала примерно:

test accuracy ≈ 0.83

А QC с data re-uploading дала:

test accuracy ≈ 0.93

Это говорит о том, что повторное кодирование признаков в каждом слое позволяет маленькой квантовой схеме строить более гибкую нелинейную решающую границу.

⸻

Структура проекта

quantum-ml-classifier/
├── README.md
├── requirements.txt
├── data/
│   └── moons.npz
├── figures/
│   ├── moons.png
│   ├── v1/
│   │   ├── loss_*_n_layers.png
│   │   └── decision_boundary_*.png
│   └── v2/
│       ├── loss_*_n_layers.png
│       └── decision_boundary_*.png
└── src/
    ├── first_circuit.py
    ├── bell_expectations.py
    ├── prepare_dataset.py
    ├── quantum_forward.py
    ├── simple_classifier_v1.py
    └── simple_classifier_v2.py

⸻

Ограничения

Проект не утверждает наличие практического квантового преимущества.

Для маленьких классических датасетов классические ML-модели, такие как SVM и нейросети, обычно быстрее, стабильнее и проще в обучении.

Цель проекта — исследовать механику смеси обучения на квантовой схеме и классической оптимизации:

Все эксперименты выполнены на statevector-симуляторе, а не на реальном шумном квантовом устройстве.