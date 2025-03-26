import matplotlib.pyplot as plt
from task1 import *
from sklearn.model_selection import train_test_split
import time


def print_table(data, headers=None, padding=2):
    """
    Prints data as a formatted table.

    Args:
        data (list of lists): The data to be printed. Each inner list represents a row.
        headers (list, optional): A list of headers for the table. Defaults to None.
        padding (int, optional): The amount of padding to add around each cell. Defaults to 2.
    """

    # If headers are provided, add them to the data
    if headers:
        data_to_print = np.row_stack((headers, data))
    else:
        data_to_print = data

    # Calculate column widths
    col_widths = [0] * len(data_to_print[0])
    for row in data_to_print:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Add padding to column widths
    col_widths = [width + 2 * padding for width in col_widths]

    # Print the table
    for row in data_to_print:
        formatted_row = ""
        for i, cell in enumerate(row):
            formatted_row += str(cell).ljust(col_widths[i])
            if i + 1 == len(row):
                formatted_row += '\\\\'
            else:
                formatted_row += ' & '

        print(formatted_row)


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
    # plt.savefig(fname="task4a_x1_trajectory", dpi=300)
    plt.show()


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

    # Evaluating trajectory quality
    trajectory_eval(model1, model2)

    # Finding optimal parameters
    model1_mses = []
    model2_mses = []
    model1_training_time = []
    model2_training_time = []
    leaf_sizes = [1, 2, 4, 6, 8, 10, 12, 14]

    print('Leaf Sizes')
    # training multiple different trees
    for leaf_size in leaf_sizes:
        model1 = RegressionTree(leaf_size=leaf_size)
        model2 = RegressionTree(leaf_size=leaf_size)

        model1_start = time.time()
        model1.fit(X, y[:, 0])
        model1_end = time.time()
        model1_training_time.append(model1_end - model1_start)

        model2_start = time.time()
        model2.fit(X, y[:, 1])
        model2_end = time.time()
        model2_training_time.append(model2_end - model2_start)

        model1_predictions, model2_predictions = make_predictions(
            model1, model2, X)

        model1_mse, model2_mse = compute_mse(
            y, model1_predictions, model2_predictions)
        model1_mses.append(model1_mse)
        model2_mses.append(model2_mse)

    headers = ["Leaf Size", "MSE", "Time Cost"]
    x1_data = np.array([leaf_sizes, model1_mses, model1_training_time]).T
    x2_data = np.array([leaf_sizes, model1_mses, model1_training_time]).T
    print_table(x1_data, headers, padding=2)
    print_table(x2_data, headers, padding=2)

    model1_mses = []
    model2_mses = []
    model1_training_time = []
    model2_training_time = []
    max_heights = [1, 2, 3, 4, 5, 6, 7, 8]

    print('Max Height')
    # training multiple different trees
    for max_height in max_heights:
        model1 = RegressionTree(max_height=max_height)
        model2 = RegressionTree(max_height=max_height)

        model1_start = time.time()
        model1.fit(X, y[:, 0])
        model1_end = time.time()
        model1_training_time.append(model1_end - model1_start)

        model2_start = time.time()
        model2.fit(X, y[:, 1])
        model2_end = time.time()
        model2_training_time.append(model2_end - model2_start)

        model1_predictions, model2_predictions = make_predictions(
            model1, model2, X)

        model1_mse, model2_mse = compute_mse(
            y, model1_predictions, model2_predictions)
        model1_mses.append(model1_mse)
        model2_mses.append(model2_mse)

    headers = ["Max Height", "MSE", "Time Cost"]
    x1_data = np.array([leaf_sizes, model1_mses, model1_training_time]).T
    x2_data = np.array([leaf_sizes, model1_mses, model1_training_time]).T
    print_table(x1_data, headers, padding=2)
    print_table(x2_data, headers, padding=2)

if __name__ == '__main__':
    main()
