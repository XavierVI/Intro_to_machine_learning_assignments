from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt

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

def plot_accuracy(model_predictions, labels, params: list[str]):
    accuracy = np.sum(np.where(model_predictions == labels, 1, 0)) \
                / y_test.shape[0]
    plt.bar(['params'], accuracy)

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


# loading the dataset and splitting the data using scikit learn
test_size = 0.3
data = np.loadtxt('./data_d10_n1000_u5.csv', delimiter=',')
X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# training each model
primal_model.fit(X_train, y_train)
dual_model.fit(X_train, y_train)

primal_predictions = primal_model.predict(X_test)
print(np.sum(np.where(primal_predictions == y_test, 1, 0)) / y_test.shape[0])

dual_model = dual_model.predict(X_test)
print(np.sum(np.where(dual_model == y_test, 1, 0)) / y_test.shape[0])

"""
NOTE: you want to plot the accuracy of the model vs. the scale combinations d and n.

One way to do this is to create one figure (a bar graph) for each scale combination, and plot the accuracy for each model.
Is this the best way to represent the data?
"""
plot_accuracy(dual_model, y_test, ['d=10, n=1,000'])
plot_accuracy(dual_model, y_test, ['d=10, n=1,000'])


