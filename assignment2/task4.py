from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt

import os

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
    

def train_models(primal_model, dual_model, X_train, y_train):
    primal_model.fit(X_train, y_train)
    dual_model.fit(X_train, y_train)


# define the models
loss = 'hinge'
random_state = 1
primal_model = LinearSVC(
    # loss=loss,
    random_state=random_state,
    dual=False,
    penalty='l2'
)

dual_model = LinearSVC(
    # loss=loss,
    random_state=random_state,
    dual=True,
    penalty='l2'
)


test_size = 0.3
data_dir = os.path.join(os.curdir, 'data')

for file in os.listdir(data_dir):
    # loading the dataset and splitting the data using scikit learn
    file_path = os.path.join(data_dir, file)
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    d = X.shape[1]
    n = X.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # training each model
    train_models(primal_model, dual_model, X_train, y_train)

    primal_predictions = primal_model.predict(X_test)
    primal_accuracy = np.sum(
        np.where(primal_predictions == y_test, 1, 0)) / y_test.shape[0] * 100

    dual_predictions = dual_model.predict(X_test)
    dual_accuracy = np.sum(
        np.where(dual_predictions == y_test, 1, 0)) / y_test.shape[0] * 100
    
    label = [f'd={d}, n={n}']
    plt.bar(label, primal_accuracy, width=0.15, color='red')
    plt.bar(label, dual_accuracy, width=0.10, color='blue')


"""
NOTE: you want to plot the accuracy of the model vs. the scale combinations d and n.

One way to do this is to create one figure (a bar graph) for each scale combination, and plot the accuracy for each model.
Is this the best way to represent the data?
"""
plt.xlabel("Scale Combinations")
plt.ylabel("Accuracy (%)")
plt.legend(['Primal', 'Dual'])
plt.title("Accuracy vs. Scale Combinations")
plt.show()

