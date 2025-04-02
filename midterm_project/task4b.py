from task1 import RegressionTree
import numpy as np
from sklearn.model_selection import train_test_split
from task4a import make_predictions, compute_mse
import matplotlib.pyplot as plt
import time

#### Functions
def func(x):
    """
    This function is pretty interesting. Since the state of the program
    is defined as the value of its variables (x and z in this case),
    we can say that the next state of the program only depends on x.
    That is, both x_{k+1} and z_{k+1} depend on x_k.

    This mostly serves as a reference.
    """
    z = 0
    for _ in range(20):
        if x > 1:
            x = 0
        else:
            x += 0.2
        z = z + x
    
    return (x, z)


def create_dataset(num_of_samples):
    x_values = np.random.default_rng(1).uniform(low=-3, high=3,  size=(num_of_samples,))
    X = np.zeros((20*num_of_samples, 2))
    y = np.zeros((20*num_of_samples, 2))
    
    # this for loop will compute
    # the next state based on x and z
    for i, x in enumerate(x_values):
        x_k = x
        z_k = 0
        for j in range(20):
            # add the current state as an input
            X[20*i+j, 0] = x_k
            X[20*i+j, 1] = z_k
            
            # compute the next state
            if x_k > 1:
                x_k = 0
            else:
                x_k += 0.2
            z_k += x_k
            
            # add the next state as a label
            y[20*i+j, 0] = x_k
            y[20*i+j, 1] = z_k

    return (X, y)



def trajectory_eval(model1, model2):
    # initial state
    x_k = 2
    z_k = 0

    # array of previous states (x, z)
    X = np.zeros((20,2))
    # array of next state (x, z)
    y = np.zeros((20, 2))

    for i in range(1, 20):
        X[i, 0] = x_k
        X[i, 1] = z_k

        # compute the next state
        if x_k > 1:
            x_k = 0
        else:
            x_k += 0.2
        z_k += x_k

        y[i, 0] = x_k
        y[i, 1] = z_k
        

    predictions_x1, predictions_x2 = make_predictions(model1, model2, X)

    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    #### Plotting actual trajectory
    axes[0].plot(range(20), y[:, 0], '-xb')
    axes[0].plot(range(20), predictions_x1[:], '-xr')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('State Value (x)')
    axes[0].set_title('Plot of x trajectory')
    axes[0].legend(['actual', 'predicted'])
    axes[0].grid(True)

    #### Plotting predicted trajectory
    axes[1].plot(range(20), y[:, 1], '-xb')
    axes[1].plot(range(20), predictions_x2[:], '-xr')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('State Value (z)')
    axes[1].set_title('Plot of z trajectory')
    axes[1].legend(['actual', 'predicted'])
    axes[1].grid(True)
    plt.savefig(fname="task4b_trajectory", dpi=300)
    # plt.show()

#### Generating training and testing data
X, y = create_dataset(100)
#print(X)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#### Training two models and evaluating their performance
model1 = RegressionTree()
model2 = RegressionTree()

model1.fit(X_train, y_train[:, 0])
model2.fit(X_train, y_train[:, 1])

x_predictions, z_predictions = make_predictions(model1, model2, X_test)

# print(f'MSE: {compute_mse(y_test, x_predictions, z_predictions)}')


#### Plotting trajectories
print('Generating trajectory plots...')
trajectory_eval(model1, model2)

# Finding optimal parameters
print(f'Building table...')
leaf_sizes = [10, 25, 50, 100, 150]
max_heights = [2, 4, 8, 16]
max_heights_table = []
leaf_sizes_table = []
model1_mses = []
model2_mses = []
model1_training_time = []
model2_training_time = []

for max_height in max_heights:
    for leaf_size in leaf_sizes:
        max_heights_table.append(max_height)
        leaf_sizes_table.append(leaf_size)
        model1 = RegressionTree(leaf_size=leaf_size, max_height=max_height)
        model2 = RegressionTree(leaf_size=leaf_size, max_height=max_height)

        # training two trees and saving the time cost
        model1_start = time.time()
        model1.fit(X_train, y_train[:, 0])
        model1_end = time.time()
        model1_training_time.append(model1_end - model1_start)

        model2_start = time.time()
        model2.fit(X_train, y_train[:, 1])
        model2_end = time.time()
        model2_training_time.append(model2_end - model2_start)

        model1_predictions, model2_predictions = make_predictions(
            model1, model2, X_test)

        model1_mse, model2_mse = compute_mse(
            y_test, model1_predictions, model2_predictions)
        model1_mses.append(model1_mse)
        model2_mses.append(model2_mse)

x1_data = np.array([max_heights_table, leaf_sizes_table,
                    model1_mses, model1_training_time]).T
x2_data = np.array([max_heights_table, leaf_sizes_table,
                    model2_mses, model2_training_time]).T

# printing out the data as a table (formatted to be copied and pasted into OverLeaf)
for row in x1_data:
    print(f'{row[0]} & {row[1]} & {row[2]:<.2e} & {row[3]:<.2e} \\\\')

print('=======================================================')

for row in x2_data:
    print(f'{row[0]} & {row[1]} & {row[2]:<.2e} & {row[3]:<.2e} \\\\')
