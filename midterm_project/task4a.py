import matplotlib.pyplot as plt
from task1 import *
from sklearn.model_selection import train_test_split
import time


def next_state(x_1, x_2):
    y1 = 0.9*x_1 - 0.2*x_2
    y2 = 0.2*x_1 + 0.9*x_2
    return np.column_stack((y1, y2))


def make_predictions(model1, model2, X):
    predictions_x1 = []
    predictions_x2 = []

    for x in X:
        predictions_x1.append(model1.predict(x))
        predictions_x2.append(model2.predict(x))

    return (predictions_x1, predictions_x2)


def compute_mse(y, x1_predictions, x2_predictions):
    y_hat = np.array([])
    y_hat = np.column_stack((x1_predictions, x2_predictions))
    x1_error = np.mean(np.square(y[:, 0] - y_hat[:, 0]))
    x2_error = np.mean(np.square(y[:, 1] - y_hat[:, 1]))
    return (x1_error, x2_error)

# Creating trajectory evaluation


def trajectory_eval(model1, model2):
    # Time steps
    T = np.arange(0, 20)

    # initial state
    x_k = np.array([0.5, 1.5])

    # creating an array of data
    # shape: (number of time steps (rows), number of entries in each row, dimension of each column (previous state and next state))
    X = np.zeros((T.size, 2, 2))

    for t in T:
        X[t, 0, :] = x_k
        # creating next state
        X[t, 1, :] = next_state(x_k[0], x_k[1])
        x_k = X[t, 1, :]

    y = X[:, 1]
    X = X[:, 0]

    predictions_x1, predictions_x2 = make_predictions(model1, model2, X)

    fig, axes = plt.subplots(1, 2, figsize=(
        24, 12))  # Adjust figsize as needed
    axes[0].plot(T, y[:, 0], '-xb')
    axes[0].plot(T, predictions_x1[:], '-xr')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('State Value')
    axes[0].set_title('Plot of x_1 trajectory')
    axes[0].legend(['actual', 'predicted'])
    axes[0].grid(True)

    axes[1].plot(T, y[:, 1], '-xb')
    axes[1].plot(T, predictions_x2[:], '-xr')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('State Value')
    axes[1].set_title('Plot of x_2 trajectory')
    axes[1].legend(['actual', 'predicted'])
    axes[1].grid(True)
    plt.savefig(fname="task4a_trajectory", dpi=300)
    # plt.show()


def create_dataset(num_of_samples):
    # creating a dataset
    X = np.random.default_rng(1).uniform(
        low=-5, high=5,  size=(num_of_samples, 2))
    y = next_state(X[:, 0], X[:, 1])
    return (X, y)

def main():
    # Creating a training and testing dataset
    X, y = create_dataset(2_000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

    # Training models
    # training two different models for each dimension and without hyperparameters
    model1 = RegressionTree()
    model1.fit(X_train, y_train[:, 0])
    # model1.bst.plot_tree()

    model2 = RegressionTree()
    model2.fit(X_train, y_train[:, 1])
    # model2.bst.plot_tree()

    # printing testing error
    model1_predictions = []
    model2_predictions = []

    for x in X_test:
        model1_predictions.append(model1.predict(x))
        model2_predictions.append(model2.predict(x))
    print(compute_mse(y_test, model1_predictions, model2_predictions))

    #### Evaluating performance on a trajectory
    trajectory_eval(model1, model2)
    # model1.bst.plot_tree()
    # model2.bst.plot_tree()

    #### Finding optimal parameters
    leaf_sizes  = [10, 25, 50]
    max_heights = [ 2, 4, 8, 16]
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

    x1_data = np.array([max_heights_table, leaf_sizes_table, model1_mses, model1_training_time]).T
    x2_data = np.array([max_heights_table, leaf_sizes_table, model2_mses, model2_training_time]).T
    
    # printing out the data as a table (formatted to be copied and pasted into OverLeaf)
    for row in x1_data:
        print(f'{row[0]} & {row[1]} & {row[2]:<.2e} & {row[3]:<.2e} \\\\')

    print('=======================================================')

    for row in x2_data:
        print(f'{row[0]} & {row[1]} & {row[2]:<.2e} & {row[3]:<.2e} \\\\')
    

if __name__ == '__main__':
    main()
