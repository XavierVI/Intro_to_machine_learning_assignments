from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt

import os
import time

"""
Read the scikit-learn documentation for SVMs. We are going to investigate the performance of
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
loss = 'squared_hinge'
random_state = 1
max_iters = 500
primal_model = LinearSVC(
    loss=loss,
    random_state=random_state,
    dual=False,
    max_iter=max_iters
)

dual_model = LinearSVC(
    loss=loss,
    random_state=random_state,
    dual=True,
    max_iter=max_iters
)


test_size = 0.3
data_dir = os.path.join(os.curdir, 'data')
primal_time_costs = []
dual_time_costs = []
primal_accuracies = []
dual_accuracies = []
scale_combinations = []


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
    primal_time_costs.append(end - start)

    start = time.time()
    dual_model.fit(X_train, y_train)
    end = time.time()
    dual_time_costs.append(end - start)

    # making predictions and calculating the accuracy for each model
    primal_predictions = primal_model.predict(X_test)
    primal_accuracy = np.sum(
        np.where(primal_predictions == y_test, 1, 0)) / y_test.shape[0] * 100
    primal_accuracies.append(primal_accuracy)

    dual_predictions = dual_model.predict(X_test)
    dual_accuracy = np.sum(
        np.where(dual_predictions == y_test, 1, 0)) / y_test.shape[0] * 100
    dual_accuracies.append(dual_accuracy)
    
    scale_combinations.append(f'd={d}, n={n}')


# creating two figures
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

x = np.arange(len(scale_combinations))  # Label positions

bar_width = 0.35  # Width of bars

ax[0].bar(x - bar_width / 2, primal_accuracies,
          bar_width, label="Primal", color="red")
ax[0].bar(x + bar_width / 2, dual_accuracies,
          bar_width, label="Dual", color="blue")

ax[0].set_xticks(x)
ax[0].set_xticklabels(scale_combinations, rotation=30, ha="right")
ax[0].set_xlabel("Scale Combinations")
ax[0].set_ylabel("Accuracy (%)")
ax[0].set_title("Accuracy vs. Scale Combinations")
ax[0].legend()

# --- Plot Time Cost ---
ax[1].bar(x - bar_width / 2, primal_time_costs,
          bar_width, label="Primal", color="red")
ax[1].bar(x + bar_width / 2, dual_time_costs,
          bar_width, label="Dual", color="blue")

ax[1].set_xticks(x)
ax[1].set_xticklabels(scale_combinations, rotation=30, ha="right")
ax[1].set_xlabel("Scale Combinations")
ax[1].set_ylabel("Training Time (s)")
ax[1].set_title("Training Time vs. Scale Combinations")
ax[1].legend()

plt.tight_layout()
plt.show()
