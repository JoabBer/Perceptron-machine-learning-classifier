import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import csv


# -----------------------------
# Data helpers
# -----------------------------
def load_iris_two_features():
    iris = load_iris()
    # x1 = sepal width (index 1), x2 = petal length (index 2)
    X = iris.data[:, [1, 2]].astype(float)
    y = iris.target.astype(int)
    names = iris.target_names
    return X, y, names


def get_class_pair(X, y, class_pos, class_neg):
    mask = (y == class_pos) | (y == class_neg)
    Xp = X[mask]
    yp = y[mask]
    # class_pos -> +1, class_neg -> -1
    t = np.where(yp == class_pos, 1, -1).astype(int)
    return Xp, t


def stratified_split_two_class(X, t, train_ratio, seed=42):
    rng = np.random.default_rng(seed)
    idx_pos = np.where(t == 1)[0]
    idx_neg = np.where(t == -1)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    n_pos_train = int(np.round(train_ratio * len(idx_pos)))
    n_neg_train = int(np.round(train_ratio * len(idx_neg)))

    train_idx = np.concatenate([idx_pos[:n_pos_train], idx_neg[:n_neg_train]])
    test_idx = np.concatenate([idx_pos[n_pos_train:], idx_neg[n_neg_train:]])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], t[train_idx], X[test_idx], t[test_idx]


def augment(X):
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack([ones, X])  # [1, x1, x2]


def build_Y_transformed(X, t):
    """
    Manual-perceptron setup:
      y_i = t_i * [1, x1, x2]
    Correct classification means: a^T y_i > 0 for all i.
    """
    Y = augment(X)
    return Y * t.reshape(-1, 1)



# Manual Perceptron Criterion GD

def perceptron_gd_train_manual(X_train, t_train, eta=0.01, theta=0.0, a0=None, max_iter=300):
    """
    Lab manual:
      misclassified set Y(a) = { y : a^T y <= 0 }
      Jp(a)  = sum_{y in Y(a)} (-a^T y)
      grad   = sum_{y in Y(a)} (-y)
      a <- a - eta*grad
    """
    Y = build_Y_transformed(X_train, t_train)

    if a0 is None:
        a = np.array([0.0, 0.0, 1.0], dtype=float)  # manual a(0)
    else:
        a = np.array(a0, dtype=float).copy()

    history = []

    for k in range(1, max_iter + 1):
        g = Y @ a
        mis_idx = np.where(g <= 0)[0]

        if mis_idx.size == 0:
            history.append({
                "iter": k,
                "Jp": 0.0,
                "grad": np.zeros(3),
                "a": a.copy(),
                "misclassified": 0
            })
            break

        Ymis = Y[mis_idx]
        Jp = float(np.sum(-(Ymis @ a)))      # should be >= 0
        grad = np.sum(-Ymis, axis=0)         # manual gradient

        history.append({
            "iter": k,
            "Jp": Jp,
            "grad": grad.copy(),
            "a": a.copy(),
            "misclassified": int(mis_idx.size)
        })

        step = eta * grad
        a = a - step

        if np.linalg.norm(step) < theta:
            break

    return a, history


def accuracy_manual(X, t, a):
    Y = build_Y_transformed(X, t)
    return float(np.mean((Y @ a) > 0))



# Task 5 plotting + table

def plot_data_and_boundary(X_train, t_train, a, title):
    """
    Boundary from raw discriminant:
      a^T [1,x1,x2] = 0 -> x2 = -(a0 + a1*x1)/a2
    """
    plt.figure()
    plt.scatter(X_train[t_train == 1, 0], X_train[t_train == 1, 1], marker='o', label='Train: class +1')
    plt.scatter(X_train[t_train == -1, 0], X_train[t_train == -1, 1], marker='x', label='Train: class -1')

    x1_min = X_train[:, 0].min() - 0.2
    x1_max = X_train[:, 0].max() + 0.2
    x1 = np.linspace(x1_min, x1_max, 200)

    a0, a1, a2 = a
    if abs(a2) < 1e-12:
        if abs(a1) > 1e-12:
            plt.axvline(x=-a0 / a1, linestyle='--', label='Decision boundary')
    else:
        x2 = -(a0 + a1 * x1) / a2
        plt.plot(x1, x2, linestyle='--', label='Decision boundary')

    plt.xlabel("x1 = sepal width")
    plt.ylabel("x2 = petal length")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_Jp(history, title):
    plt.figure()
    iters = [row["iter"] for row in history]
    Jps = [row["Jp"] for row in history]
    plt.plot(iters, Jps)
    plt.xlabel("Iteration")
    plt.ylabel("Jp(a)")
    plt.title(title)
    plt.grid(True)
    plt.show()


def save_iteration_table(history, filename):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "Jp", "grad0", "grad1", "grad2", "a0", "a1", "a2", "misclassified"])
        for row in history:
            g0, g1, g2 = row["grad"]
            a0, a1, a2 = row["a"]
            w.writerow([row["iter"], row["Jp"], g0, g1, g2, a0, a1, a2, row["misclassified"]])


# -----------------------------
# Runs
# -----------------------------
def run_no_plots(class_pos, class_neg, train_ratio, eta=0.01, theta=0.0, a_init=(0, 0, 1), max_iter=300, seed=42):
    X, y, names = load_iris_two_features()
    Xp, t = get_class_pair(X, y, class_pos, class_neg)
    Xtr, ttr, Xte, tte = stratified_split_two_class(Xp, t, train_ratio=train_ratio, seed=seed)

    a, history = perceptron_gd_train_manual(Xtr, ttr, eta=eta, theta=theta, a0=a_init, max_iter=max_iter)

    acc_train = accuracy_manual(Xtr, ttr, a)
    acc_test = accuracy_manual(Xte, tte, a)

    pair_name = f"{names[class_pos]}(+1) vs {names[class_neg]}(-1)"
    split_name = f"{int(train_ratio*100)}% train / {int((1-train_ratio)*100)}% test"

    print("\n==============================")
    print(f"PAIR: {pair_name}")
    print(f"SPLIT: {split_name}")
    print(f"eta={eta}, theta={theta}, a_init={list(a_init)}, max_iter={max_iter}")
    print(f"Final a = {np.round(a, 4)}")
    print(f"Iterations used = {history[-1]['iter'] if history else 0}")
    print(f"Train accuracy = {acc_train*100:.2f}%")
    print(f"Test  accuracy = {acc_test*100:.2f}%")
    print("==============================")

    return a, history, (Xtr, ttr)


def run_task5_two_cases(class_pos, class_neg, train_ratio, cases, theta=0.0, max_iter=300, seed=42):
    """
    cases: list of exactly 2 dicts like:
      {"eta": 0.01, "a_init": (0,0,1), "tag": "case1"}
      {"eta": 0.1,  "a_init": (0,0,1), "tag": "case2"}
    """
    X, y, names = load_iris_two_features()
    Xp, t = get_class_pair(X, y, class_pos, class_neg)
    Xtr, ttr, Xte, tte = stratified_split_two_class(Xp, t, train_ratio=train_ratio, seed=seed)

    pair_name = f"{names[class_pos]}(+1) vs {names[class_neg]}(-1)"
    split_name = f"{int(train_ratio*100)}% train / {int((1-train_ratio)*100)}% test"

    for c in cases:
        eta = c["eta"]
        a_init = c["a_init"]
        tag = c["tag"]

        a, history = perceptron_gd_train_manual(Xtr, ttr, eta=eta, theta=theta, a0=a_init, max_iter=max_iter)
        acc_train = accuracy_manual(Xtr, ttr, a)
        acc_test = accuracy_manual(Xte, tte, a)

        print("\n----- TASK 5 CASE -----")
        print(f"Case tag: {tag}")
        print(f"{pair_name} | {split_name}")
        print(f"eta={eta}, theta={theta}, a_init={list(a_init)}, max_iter={max_iter}")
        print(f"Final a = {np.round(a, 4)}")
        print(f"Iterations used = {history[-1]['iter'] if history else 0}")
        print(f"Train accuracy = {acc_train*100:.2f}%")
        print(f"Test  accuracy = {acc_test*100:.2f}%")

        #  PLOTS FOR TASK 5
        plot_data_and_boundary(
            X_train=Xtr, t_train=ttr, a=a,
            title=f"Task 5 | {pair_name} | {split_name} | eta={eta} | a0={list(a_init)}"
        )
        plot_Jp(
            history,
            title=f"Task 5 | Jp(a) over iterations | {pair_name} | eta={eta} | a0={list(a_init)}"
        )

        # TABLE FOR TASK 5
        csv_name = f"task5_iter_table_{tag}.csv"
        save_iteration_table(history, csv_name)
        print(f"Saved: {csv_name}")


def main():
    # sklearn class indices: 0=setosa(A), 1=versicolor(B), 2=virginica(C)

  
    # Tasks 1-4: 
    # Task 1-3: A vs B
    run_no_plots(class_pos=0, class_neg=1, train_ratio=0.30, eta=0.01, theta=0.0, a_init=(0, 0, 1), max_iter=300, seed=42)
    run_no_plots(class_pos=0, class_neg=1, train_ratio=0.70, eta=0.01, theta=0.0, a_init=(0, 0, 1), max_iter=300, seed=42)

    # Task 4: B vs C
    run_no_plots(class_pos=1, class_neg=2, train_ratio=0.30, eta=0.01, theta=0.0, a_init=(0, 0, 1), max_iter=300, seed=42)
    run_no_plots(class_pos=1, class_neg=2, train_ratio=0.70, eta=0.01, theta=0.0, a_init=(0, 0, 1), max_iter=300, seed=42)

    # -------------------------
    # Task 5: ONLY 2 CASES WITH DIFFERENT STEP SIZES
    # -------------------------
   #Task 5 to use A vs B instead, change class_pos=1, class_neg=2 to class_pos=0, class_neg=1.
    task5_cases = [
        {"eta": 0.005, "a_init": (0, 0, 1), "tag": "eta0p005_a001"},
        {"eta": 0.05,  "a_init": (0, 0, 1), "tag": "eta0p05_a001"},
    ]

    run_task5_two_cases(
        class_pos=1, class_neg=2,     # B vs C
        train_ratio=0.70,             # 70% training ratio, can change to 30% 
        cases=task5_cases,
        theta=0.0,
        max_iter=300,
        seed=42
    )


if __name__ == "__main__":
    main()