import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# ELE888 Lab 3
# 2-2-1 Multilayer Neural Network for XOR using batch backprop
# Activation: tanh
# Targets: -1 / +1
# =========================================================
 # random value from numpy collection but I want to repeat the 
 # inital first guess everytime I run it for repeatability
np.random.seed(42) 


# -----------------------------
# Activation function and derivative
# -----------------------------
def tanh(x):
    return np.tanh(x)


def tanh_derivative_from_output(y):
    """
    y = tanh(net), derivative is:
    f'(net) = 1 - y^2
    """
    return 1.0 - y**2


# -----------------------------
# XOR data from lab sheet
# -----------------------------
X = np.array([
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1]
], dtype=float)

T = np.array([
    [-1],
    [ 1],
    [ 1],
    [-1]
], dtype=float)

# -----------------------------
# Hyperparameters
# -----------------------------
eta = 0.1 #step value for gradient descent
theta = 0.001 #halt condition for training: stop when SSE < theta
max_epochs = 50000 # maximum number of epochs to prevent infinite loop if not converging


# -----------------------------
# Network architecture: 2-2-1
# Include bias using separate vectors
# W_input_hidden: shape (2 hidden, 2 input)
# b_hidden: shape (2,)
# W_hidden_output: shape (1 output, 2 hidden)
# b_output: shape (1,)
# -----------------------------
#the inital weight guess is randomly generated between -1 and 1,
#  2 hidden neurons and 2 input features 
W_input_hidden = np.random.uniform(-1, 1, (2, 2)) 
b_hidden = np.random.uniform(-1, 1, (2,))

W_hidden_output = np.random.uniform(-1, 1, (1, 2))
b_output = np.random.uniform(-1, 1, (1,))

# Store learning curve
error_history = []


# -----------------------------
# Forward pass for one sample
# means to compute net and output for hidden and output layers in the 
#regular forward direction, we will use this function to compute the forward pass 
# for each sample during training and also for final verification after training
# -----------------------------
def forward_one(x):
    # Hidden layer
    net_h = W_input_hidden @ x + b_hidden          # shape (2,) this means 
    y_h = tanh(net_h)                              # shape (2,)

    # Output layer
    net_o = W_hidden_output @ y_h + b_output       # shape (1,)
    y_o = tanh(net_o)                              # shape (1,)

    return net_h, y_h, net_o, y_o


# -----------------------------
# Training: batch backpropagation
#batch backpropagation means we will accumulate the weight updates for all samples in
#  one epoch (one pass through the entire training set) and then update the weights after
#  processing the entire batch, this is different from online (stochastic) backprop where we
#  update weights after each sample

# -----------------------------
for epoch in range(max_epochs):
    # Batch accumulators
    dW_ih = np.zeros_like(W_input_hidden) #derivative of weights from input to hidden layer
    db_h = np.zeros_like(b_hidden) #derivative of bias for hidden layer

    dW_ho = np.zeros_like(W_hidden_output) #derivative of weights from hidden to output layer
    db_o = np.zeros_like(b_output) #derivative of bias for output layer

    sse = 0.0

    for x, t in zip(X, T):
        # ---------- forward ----------
        net_h, y_h, net_o, y_o = forward_one(x)

        # ---------- error ----------
        error = t - y_o #real output - prediction error for the output layer 
        sse += np.sum(error**2) # squared error

        # ---------- sensitivities ----------
        # output sensitivity: delta_k = f'(net_k) * (t_k - y_k)
        delta_o = tanh_derivative_from_output(y_o) * error    #output layer sensitivity     # shape (1,)

        # hidden sensitivity: delta_j = f'(net_j) * sum_k(w_kj * delta_k)
        delta_h = tanh_derivative_from_output(y_h) * (W_hidden_output.T @ delta_o).flatten()  # shape (2,)

        # ---------- batch accumulate ----------
        # Hidden -> Output
        dW_ho += eta * np.outer(delta_o, y_h)   # shape (1,2)
        db_o += eta * delta_o.flatten()

        # Input -> Hidden
        dW_ih += eta * np.outer(delta_h, x)      # shape (2,2)
        db_h += eta * delta_h

    # ---------- batch (weights and biases) update ----------
    W_input_hidden += dW_ih
    b_hidden += db_h

    W_hidden_output += dW_ho
    b_output += db_o

    error_history.append(sse)

    if sse < theta:
        print(f"Converged at epoch {epoch + 1}")
        break
else:
    print(f"Did not reach theta = {theta} within {max_epochs} epochs")


# -----------------------------
# Final verification on XOR
# -----------------------------
print("\nFinal weights and biases:")
print("W_input_hidden =\n", W_input_hidden)
print("b_hidden =\n", b_hidden)
print("W_hidden_output =\n", W_hidden_output)
print("b_output =\n", b_output)

print("\nXOR verification:")
print(" x1   x2    target    y_raw      predicted_class")

hidden_outputs = []

for x, t in zip(X, T):
    net_h, y_h, net_o, y_o = forward_one(x)
    hidden_outputs.append(y_h)

    predicted_class = 1 if y_o[0] >= 0 else -1
    print(f"{int(x[0]):>3} {int(x[1]):>4} {int(t[0]):>8} {y_o[0]:>10.5f} {predicted_class:>16}")

hidden_outputs = np.array(hidden_outputs)


# -----------------------------
# Plot 1: learning curve
# -----------------------------
plt.figure(figsize=(7, 5))
plt.plot(error_history)
plt.xlabel("Epoch")
plt.ylabel("Sum Squared Error (SSE)")
plt.title("Learning Curve for XOR (2-2-1 MNN)")
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# Plot 2: decision surface in x1-x2 space
# -----------------------------
x1_vals = np.linspace(-1.5, 1.5, 250)
x2_vals = np.linspace(-1.5, 1.5, 250)
xx1, xx2 = np.meshgrid(x1_vals, x2_vals)

zz = np.zeros_like(xx1)

for i in range(xx1.shape[0]):
    for j in range(xx1.shape[1]):
        x_grid = np.array([xx1[i, j], xx2[i, j]])
        _, _, _, y_grid = forward_one(x_grid)
        zz[i, j] = y_grid[0]

plt.figure(figsize=(7, 6))
plt.contourf(xx1, xx2, zz, levels=50, alpha=0.8, cmap="coolwarm")
plt.contour(xx1, xx2, zz, levels=[0], linewidths=2)  # decision boundary y=0

# plot training points
for x, t in zip(X, T):
    marker = "o" if t[0] == 1 else "s"
    plt.scatter(x[0], x[1], marker=marker, s=120, edgecolors="k")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Decision Surface in Input Space (x1-x2)")
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# Plot 3: decision surface in hidden space (y1-y2)
# -----------------------------
# For the hidden space, output neuron receives (y1, y2)
y1_vals = np.linspace(-1.1, 1.1, 250)
y2_vals = np.linspace(-1.1, 1.1, 250)
yy1, yy2 = np.meshgrid(y1_vals, y2_vals)

zz_hidden = np.zeros_like(yy1)

for i in range(yy1.shape[0]):
    for j in range(yy1.shape[1]):
        hidden_vec = np.array([yy1[i, j], yy2[i, j]])
        net_o = W_hidden_output @ hidden_vec + b_output
        y_o = tanh(net_o)
        zz_hidden[i, j] = y_o[0]

plt.figure(figsize=(7, 6))
plt.contourf(yy1, yy2, zz_hidden, levels=50, alpha=0.8, cmap="coolwarm")
plt.contour(yy1, yy2, zz_hidden, levels=[0], linewidths=2)  # boundary in hidden space

# plot actual transformed XOR points
for y_h, t in zip(hidden_outputs, T):
    marker = "o" if t[0] == 1 else "s"
    plt.scatter(y_h[0], y_h[1], marker=marker, s=120, edgecolors="k")

plt.xlabel("y1")
plt.ylabel("y2")
plt.title("Decision Surface in Hidden Space (y1-y2)")
plt.grid(True)
plt.tight_layout()
plt.show()