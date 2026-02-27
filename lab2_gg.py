import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# ==========================================
# Data Preparation
# ==========================================
# Load IRIS dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Extract features: x1 (sepal width, index 1) and x2 (petal length, index 2) [cite: 46]
# Data set A: Setosa (class 0), Data set B: Versicolour (class 1), Data set C: Virginica (class 2) [cite: 46, 47, 48]
X_A = X[y == 0][:, [1, 2]]
X_B = X[y == 1][:, [1, 2]]
X_C = X[y == 2][:, [1, 2]]

# Create augmented feature vectors (adding x0 = 1) [cite: 23]
X_A_aug = np.hstack((np.ones((X_A.shape[0], 1)), X_A))
X_B_aug = np.hstack((np.ones((X_B.shape[0], 1)), X_B))
X_C_aug = np.hstack((np.ones((X_C.shape[0], 1)), X_C))

# ==========================================
# Gradient Descent Implementation
# ==========================================
def perceptron_gradient_descent(X1, X2, a_init, eta, max_iter=300, theta=0):
    """
    Computes weight vector using gradient descent and perceptron criterion.
    X1: Class 1 augmented features
    X2: Class 2 augmented features
    """
    # Standardize to a^T * y <= 0 for misclassification by negating Class 2
    Y1 = X1
    Y2 = -X2
    Y = np.vstack((Y1, Y2))
    
    a = np.array(a_init, dtype=float)
    history = []
    
    for k in range(max_iter):
        # Find misclassified samples Y(a) where a^T * y <= 0 [cite: 52]
        predictions = np.dot(Y, a)
        misclassified_indices = np.where(predictions <= 0)[0]
        Y_misclassified = Y[misclassified_indices]
        
        # Compute Criterion Function J_p(a) and Gradient [cite: 51]
        if len(Y_misclassified) > 0:
            J_p = np.sum(-np.dot(Y_misclassified, a))
            grad_J_p = np.sum(-Y_misclassified, axis=0)
        else:
            J_p = 0
            grad_J_p = np.zeros_like(a)
            
        history.append({
            'iteration': k+1,
            'J_p': J_p,
            'grad_J_p': grad_J_p.copy(),
            'a': a.copy()
        })
        
        # Stop condition [cite: 35]
        if np.linalg.norm(eta * grad_J_p) <= theta:
            break
            
        # Update weights: a = a - eta * grad_J_p [cite: 34]
        a = a - eta * grad_J_p
        
    return a, history

def calculate_accuracy(X1, X2, a):
    correct_1 = np.sum(np.dot(X1, a) > 0)
    correct_2 = np.sum(np.dot(X2, a) < 0)
    total = X1.shape[0] + X2.shape[0]
    return (correct_1 + correct_2) / total * 100

def run_task(name, X_class1, X_class2, train_ratio, a_init, eta):
    print(f"\n--- {name} (Train: {train_ratio*100}%) ---")
    
    # Train/Test Split
    X1_train, X1_test = train_test_split(X_class1, train_size=train_ratio, random_state=42)
    X2_train, X2_test = train_test_split(X_class2, train_size=train_ratio, random_state=42)
    
    # Run gradient descent [cite: 57, 58]
    a_final, history = perceptron_gradient_descent(X1_train, X2_train, a_init, eta)
    
    # Compute testing accuracy
    acc = calculate_accuracy(X1_test, X2_test, a_final)
    
    print(f"Final Weights: {a_final}")
    print(f"Iterations to converge: {len(history)}")
    print(f"Testing Classification Accuracy: {acc:.2f}%")
    return a_final, history, X1_train, X2_train

# ==========================================
# Tasks 1 to 4 [cite: 56, 59, 60, 61]
# ==========================================
a_init_std = [0, 0, 1]
eta_std = 0.01

run_task("Tasks 1 & 2: Dataset A vs B", X_A_aug, X_B_aug, 0.30, a_init_std, eta_std)
run_task("Task 3: Dataset A vs B", X_A_aug, X_B_aug, 0.70, a_init_std, eta_std)
run_task("Task 4a: Dataset B vs C", X_B_aug, X_C_aug, 0.30, a_init_std, eta_std)
run_task("Task 4b: Dataset B vs C", X_B_aug, X_C_aug, 0.70, a_init_std, eta_std)

# ==========================================
# Task 5: Different Parameters Study [cite: 62]
# ==========================================
print("\n--- Task 5: Custom Configurations (Dataset A vs B, 70% Train) ---")
configs = [
    {'eta': 0.05, 'a_init': [1, 1, 1]},
    {'eta': 0.001, 'a_init': [-1, 0, 0]}
]

for idx, config in enumerate(configs):
    eta = config['eta']
    a_init = config['a_init']
    
    a_final, history, X1_train, X2_train = run_task(
        f"Custom Config {idx+1} (eta={eta}, a_init={a_init})", 
        X_A_aug, X_B_aug, 0.70, a_init, eta
    )
    
    # Print Iteration Table [cite: 64]
    print(f"\nIteration Table (eta={eta}, a_init={a_init}):")
    print(f"{'Iter':<5} | {'J_p':<8} | {'grad_J_p (a0, a1, a2)':<25} | {'a (a0, a1, a2)'}")
    print("-" * 70)
    for h in history[:5]: # Showing first 5 to save space
        print(f"{h['iteration']:<5} | {h['J_p']:<8.2f} | {str(np.round(h['grad_J_p'], 2)):<25} | {str(np.round(h['a'], 2))}")
    
    # Plotting training data and decision boundary [cite: 65, 66]
    plt.figure(figsize=(8, 6))
    plt.scatter(X1_train[:, 1], X1_train[:, 2], label='Class 1', color='blue', marker='o')
    plt.scatter(X2_train[:, 1], X2_train[:, 2], label='Class 2', color='red', marker='x')
    
    x1_vals = np.array([X[:, 1].min() - 0.5, X[:, 1].max() + 0.5])
    if a_final[2] != 0:
        x2_vals = -(a_final[0] + a_final[1] * x1_vals) / a_final[2]
        plt.plot(x1_vals, x2_vals, 'k--', label='Decision Boundary')
    
    plt.xlabel('Feature x1 (Sepal Width)')
    plt.ylabel('Feature x2 (Petal Length)')
    plt.title(f'Decision Boundary: eta={eta}, a(0)={a_init}')
    plt.legend()
    plt.grid(True)
    plt.show()