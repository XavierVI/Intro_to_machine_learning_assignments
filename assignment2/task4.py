from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np

import os
import time

"""
Task 4 instructions:

- Read the scikit-learn documentation for SVMs. We are going to investigate the performance of
solving primal and dual problems in linear classification.

- You may use the same datasets generated in the
previous task. The easiest way to reuse a dataset is to keep all data in a file.

- In this task, you should use the built-in class LinearSVC in sklearn.svm.

- You may use the hinge loss and the default value for the
regularization parameter.

- For each scale combination, e.g., d = 500, n = 1000, compare the time costs and
prediction accuracies (on the test dataset) of training a linear SVC by solving the primal and dual problems
respectively.
"""


# define the models
random_state = 1
max_iters = 5_000_000
primal_model = LinearSVC(
    random_state=random_state,
    dual=False,
    max_iter=max_iters
)

dual_model = LinearSVC(
    random_state=random_state,
    dual=True,
    max_iter=max_iters
)

test_size = 0.3
data_dir = os.path.join(os.curdir, 'data')
table = []

for file in os.listdir(data_dir):
    # loading the dataset and splitting the data using scikit learn
    file_path = os.path.join(data_dir, file)
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    d = X.shape[1]
    n = X.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        test_size=test_size, random_state=random_state)

    # training each model and timing the training time
    start = time.time()
    primal_model.fit(X_train, y_train)
    end = time.time()
    primal_time_cost = end - start

    start = time.time()
    dual_model.fit(X_train, y_train)
    end = time.time()
    dual_time_cost = end - start

    # making predictions and calculating the accuracy for each model
    primal_predictions = primal_model.predict(X_test)
    primal_accuracy = np.mean(primal_predictions == y_test) * 100

    dual_predictions = dual_model.predict(X_test)
    dual_accuracy = np.mean(dual_predictions == y_test) * 100
    
    table.append([d, n, primal_time_cost, dual_time_cost,
                 primal_accuracy, dual_accuracy])


# sort the table
table = sorted(table, key=lambda x: (x[0], x[1]))

# printing out the data as a table (formatted to be copied and pasted into OverLeaf)
for row in table:
    print(f'{row[0]} & {row[1]} & {row[2]:<.3f} & {row[3]:<.3f} & {row[4]:<.2f} & {row[5]:<.2f} \\\\')

