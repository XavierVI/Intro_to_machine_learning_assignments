from task1 import *

# Generating data
T = np.arange(0, 20)

# Initial state 
x_k = np.array([2.0, 0.0])

# creating an array of data
# shape: (number of time steps (rows), 
#         number of entries in each row, 
#         dimension of each column (previous state and next state))
X = np.zeros((T.size, 2, 2))


for t in T:

    # Current state
    X[t, 0, :] = x_k

    # Createing next state
    if x_k[0] > 1:
        x_next = 0
    else:
        x_next = x_k[0] + 0.2

    z_next = x_k[1] + x_next

    X[t, 1, :] = np.array([x_next, z_next])
    x_k = X[t, 1, :]  # Update state for next iteration


# Define X (current states) and y (next states)
y = X[:, 1]
X = X[:, 0]

# Fit regression tree
model = RegressionTree()
model.fit(X, y[:, 0])  # Predict next x given (x, z)
model.bst.plot_tree()