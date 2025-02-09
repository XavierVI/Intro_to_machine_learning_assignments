"""
(4 points) In the textbook, both SGD and mini-batch GD are explained. Let us write a new training function
fit mini batch SGD in the class AdalineSGD such that the new function combines SGD and mini-batch
GD in training. In every epoch, the mini-batch SGD method updates the learning parameters based on a
randomly selected subset of the training data. The size (number of samples) of the subset is called the batch
size which is a hyperparameter decided by the user.
"""