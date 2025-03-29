from task1 import RegressionTree
import numpy as np
from sklearn.model_selection import train_test_split
from task4a import make_predictions, compute_mse
import matplotlib.pyplot as plt
import time

def sine(X):
    return 0.8 * np.sin(X - 1)

def make_predictions(model, X):
    predictions = np.empty((X.shape[0],))
    for i, x in enumerate(X):
        predictions[i] = model.predict(x)
    return predictions

def compute_mse(y, y_hat):
    return np.mean(np.square(y - y_hat))


#### Generating training and testing data
X = np.random.default_rng(1).uniform(-3, 3, size=(100,))
y = sine(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

#### Training a model with no hyperparameters
model = RegressionTree()
train_start = time.time()
model.fit(X_train, y_train)
train_end = time.time()
model_training_time = train_end - train_start
predictions = make_predictions(model, X_test)
model_mse  = compute_mse(y_test, predictions)

print(f'Tree height with no hyperparameters: {model.bst.get_height(model.bst.root_node)}')
print(f'None & {model_mse:<.2e} & {model_training_time:<.2e} \\\\')

#### Testing different parameters
leaf_sizes = np.array([2, 4, 8])
max_heights = np.array([6, 9])

model_mses = []
model_training_times = []

for max_height in max_heights:
    model = RegressionTree(max_height=max_height)
    train_start = time.time()
    model.fit(X_train, y_train)
    train_end = time.time()
    model_training_times.append(train_end - train_start)
    predictions = make_predictions(model, X_test)
    model_mses.append(compute_mse(y_test, predictions))

model_data = np.array([max_heights, model_mses, model_training_times]).T

# printing out the data as a table (formatted to be copied and pasted into OverLeaf)
for row in model_data:
    print(f'Max height = {row[0]} & {row[1]:<.2e} & {row[2]:<.2e} \\\\')


model_mses = []
model_training_times = []

for leaf_size in leaf_sizes:
    model = RegressionTree(leaf_size=leaf_size)
    train_start = time.time()
    model.fit(X_train, y_train)
    train_end = time.time()
    model_training_times.append(train_end - train_start)
    predictions = make_predictions(model, X_test)
    model_mses.append(compute_mse(y_test, predictions))

model_data = np.array([leaf_sizes, model_mses, model_training_times]).T

for row in model_data:
    print(f'Leaf size = {row[0]} & {row[1]:<.2e} & {row[2]:<.2e} \\\\')


