import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


# Load IRIS dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Extract features: x1 (sepal width, index 1) and x2 (petal length, index 2)
# Data set A: Setosa (class 0), Data set B: Versicolour (class 1), Data set C: Virginica (class 2)
X_A = X[y == 0][:, [1, 2]] #setpsa
X_B = X[y == 1][:, [1, 2]] #versicolour
X_C = X[y == 2][:, [1, 2]] #virginica

# Create augmented feature vectors (adding x0 = 1)
X_A_aug = np.hstack((np.ones((X_A.shape[0], 1)), X_A))
X_B_aug = np.hstack((np.ones((X_B.shape[0], 1)), X_B))
X_C_aug = np.hstack((np.ones((X_C.shape[0], 1)), X_C))


# Gradient Descent Implementation

def perceptron_gradient_descent(X1, X2, a_init, eta, max_iter=300, theta=1e-4):
    """
    Computes weight vector using gradient descent and perceptron criterion.
    X1: Class 1 augmented features
    X2: Class 2 augmented features
    theta: Small tolerance range to stop execution when gradient is close to 0
    """
    # Standardize to a^T * y <= 0 for misclassification by negating Class 2
    Y1 = X1
    Y2 = -X2
    Y = np.vstack((Y1, Y2))
    
    a = np.array(a_init, dtype=float)
    history = []
    
    for k in range(max_iter):
        # Find misclassified samples Y(a) where a^T * y <= 0
        predictions = np.dot(Y, a)
        misclassified_indices = np.where(predictions <= 0)[0]
        Y_misclassified = Y[misclassified_indices]
        
        # Compute Criterion Function J_p(a) and Gradient
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
        
        # Stop condition with tolerance range: |eta * grad_J(a)| <= theta
        if np.linalg.norm(eta * grad_J_p) <= theta:
            break
            
        # Update weights: a = a - eta * grad_J_p
        a = a - eta * grad_J_p
        
    return a, history

def calculate_accuracy(X1, X2, a):
    correct_1 = np.sum(np.dot(X1, a) > 0)
    correct_2 = np.sum(np.dot(X2, a) < 0)
    total = X1.shape[0] + X2.shape[0]
    return (correct_1 + correct_2) / total * 100

def run_task(name, X_class1, X_class2, train_ratio, a_init, eta, theta, dataset_label_1, dataset_label_2):
    print(f"\n--- {name} (Train: {train_ratio*100}%) ---")
    
    # Train/Test Split
    X1_train, X1_test = train_test_split(X_class1, train_size=train_ratio, random_state=42)
    X2_train, X2_test = train_test_split(X_class2, train_size=train_ratio, random_state=42)
    
    # Run gradient descent (limit to 300 iterations as per lab manual)
    a_final, history = perceptron_gradient_descent(X1_train, X2_train, a_init, eta, max_iter=300, theta=theta)
    
    # Compute testing accuracy
    acc = calculate_accuracy(X1_test, X2_test, a_final)
    
    print(f"Final Weights: {np.round(a_final, 4)}")
    print(f"Iterations to converge: {len(history)}")
    print(f"Testing Classification Accuracy: {acc:.2f}%")
    
    # ------------------------------------------
    # Plotting Logic Added for Every Task
    # ------------------------------------------
    plt.figure(figsize=(8, 6))
    
    # Scatter training data [cite: 65]
    plt.scatter(X1_train[:, 1], X1_train[:, 2], label=f'{dataset_label_1} (Train)', color='blue', marker='o')
    plt.scatter(X2_train[:, 1], X2_train[:, 2], label=f'{dataset_label_2} (Train)', color='red', marker='x')
    
    # Determine bounds for the decision line based on the combined data
    X_combined = np.vstack((X_class1, X_class2))
    x1_min, x1_max = X_combined[:, 1].min() - 0.5, X_combined[:, 1].max() + 0.5
    x1_vals = np.array([x1_min, x1_max])
    
    # Decision boundary: a0 + a1*x1 + a2*x2 = 0  =>  x2 = -(a0 + a1*x1) / a2 [cite: 66]
    if a_final[2] != 0:
        x2_vals = -(a_final[0] + a_final[1] * x1_vals) / a_final[2]
        plt.plot(x1_vals, x2_vals, 'k--', label='Decision Boundary')
    else:
        # Failsafe just in case a2 becomes 0 (vertical line)
        x1_vert = -a_final[0] / a_final[1]
        plt.axvline(x=x1_vert, color='k', linestyle='--', label='Decision Boundary')
    
    plt.xlabel('Feature x1 (Sepal Width)')
    plt.ylabel('Feature x2 (Petal Length)')
    plt.title(f'{name} | {dataset_label_1} vs {dataset_label_2}\nTrain: {train_ratio*100}% | eta: {eta} | a(0): {a_init}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return a_final, history

# ==========================================
# Running All Tasks
# ==========================================
a_init_std = [0, 0, 1]
eta_std = 0.01
theta_std = 0  # Strict zero to follow the manual for tasks 1-4 [cite: 57]

# Tasks 1 & 2: Dataset A vs B, 30% Train [cite: 56, 57, 59]
run_task("Tasks 1 & 2", X_A_aug, X_B_aug, 0.30, a_init_std, eta_std, theta_std, "Setosa (A)", "Versicolour (B)")

# Task 3: Dataset A vs B, 70% Train [cite: 60]
run_task("Task 3", X_A_aug, X_B_aug, 0.70, a_init_std, eta_std, theta_std, "Setosa (A)", "Versicolour (B)")

# Task 4a: Dataset B vs C, 30% Train [cite: 61]
run_task("Task 4a", X_B_aug, X_C_aug, 0.30, a_init_std, eta_std, theta_std, "Versicolour (B)", "Virginica (C)")

# Task 4b: Dataset B vs C, 70% Train [cite: 61]
run_task("Task 4b", X_B_aug, X_C_aug, 0.70, a_init_std, eta_std, theta_std, "Versicolour (B)", "Virginica (C)")

# Task 5: Different Parameters Study [cite: 62]
print("\n--- Task 5: Custom Configurations ---")
configs = [
    {'eta': 0.1, 'a_init': [1, 1, 1], 'theta': 1e-4},
    {'eta': 0.001, 'a_init': [-1, 0, 0], 'theta': 1e-4}
]

for idx, config in enumerate(configs):
    eta = config['eta']
    a_init = config['a_init']
    theta_custom = config['theta']
    
    a_final, history = run_task(
        f"Task 5 (Config {idx+1})", 
        X_A_aug, X_B_aug, 0.70, a_init, eta, theta_custom, 
        "Setosa (A)", "Versicolour (B)"
    )
    
    # Print Iteration Table for Task 5 [cite: 64]
    print(f"\nIteration Table (eta={eta}, a_init={a_init}):")
    print(f"{'Iter':<5} | {'J_p':<8} | {'grad_J_p (a0, a1, a2)':<25} | {'a (a0, a1, a2)'}")
    print("-" * 70)
    for h in history[:5]: # Showing first 5
        print(f"{h['iteration']:<5} | {h['J_p']:<8.2f} | {str(np.round(h['grad_J_p'], 2)):<25} | {str(np.round(h['a'], 2))}")
    if len(history) > 5:
        print("...")
        h = history[-1] # Showing the final iteration
        print(f"{h['iteration']:<5} | {h['J_p']:<8.2f} | {str(np.round(h['grad_J_p'], 2)):<25} | {str(np.round(h['a'], 2))}")