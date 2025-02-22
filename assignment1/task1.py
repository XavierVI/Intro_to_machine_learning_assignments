from textbook_code import Perceptron, AdalineGD
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

"""
(6 points) Use the Python programs for perceptron learning and Adaline given in the textbook and do
the comparisons on the loss, number of updates, margin of the resulting separation hyperplane, and the
convergence. What are the conclusions you can get? Can you provide any mathematical explanations to
support your conclusions? You may use the Iris dataset (the first 2 classes with all features) in the test
and are required to give at least 3 mathematically different conclusions. In order to make apple-to-apple
comparisons, you should use the same initial parameter values, training data, learning rate and number of
epochs for both perceptron learning and Adaline.

"""

epochs = 30
learning_rate = 0.01
random_state = 1


X, y = load_iris(return_X_y=True)

mask = np.isin(y, [0, 1])

X = X[mask]
y = y[mask]

print(X.shape)

perceptron = Perceptron(
    n_iter=epochs,
    eta=learning_rate,
    random_state=random_state
)
adaline_GD = AdalineGD(
    n_iter=epochs,
    eta=learning_rate,
    random_state=random_state
)

perceptron.fit(X, y)
adaline_GD.fit(X, y)


perceptron_errors = perceptron.errors_
adaline_losses = adaline_GD.losses_

plt.plot(np.arange(1, 31), perceptron_errors, '-ob')
plt.plot(np.arange(1, 31), adaline_losses, '-og')
plt.legend(['Percepton Error', 'Adaline Loss'])
plt.xlabel('Epoch')
plt.ylabel('Error / Loss')
plt.title('Comparison of Percetron and Adaline GD Performance')
# plt.show()
plt.savefig(fname='task1_convergence_plot')
