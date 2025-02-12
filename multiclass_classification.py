from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from textbook_code import Perceptron


"""
Task 4:
(4 points) Modify the class Perceptron given in the textbook such that the bias data field b is absorbed by
the weight vector w . Your program is required to be compatible with the training program in the textbook.

"""

class ModifiedPerceptron:
    """Perceptron classifier.
    
    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight 
      initialization.
    
    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
      Bias unit after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.
    
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of 
          examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        
        Returns
        -------
        self : object
        
        """
        rgen = np.random.RandomState(self.random_state)
        # initialize weights and the bias
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=X.shape[1])
        # add the bias to the end of the weight matrix
        self.w_ = np.hstack((self.w_, np.array([0])))
        # remove the bias and add an extra computation
        # self.b_ = np.float_(0.)

        # a list to accumulate the errors
        self.errors_ = []

        print(f'Started training with X: {X.shape}')
        
        for _ in range(self.n_iter):
            errors = 0
            # for each (x[i], y[i]) in (X, y)
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                # update weights
                self.w_ += update * np.concatenate((xi, [1]))
                # calculate the error (only 0 or 1)
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        
        # if X is a 1D array, add a 1 to the end
        if len(X.shape) == 1:
            X_modified = np.concatenate((X, [1]))
        else:
            # otherwise, add a ones to the end of each row
            X_modified = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

        return np.dot(X_modified, self.w_)
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)


"""
Task 3:
(6 points) A perceptron can only be used for binary classification, however, the Iris dataset has 3 classes:
setosa, versicolor and virginica. If you are only allowed to use perceptrons but the number is not limited,
how would you like to perform a multiclass classification for the Iris data set (all features)? Please write a
program (demo) for this task using the new perceptron class developed in the previous task.

"""

class MultiplePerceptrons:

    def __init__(self):
        self.classifier1 = ModifiedPerceptron()
        self.classifier2 = ModifiedPerceptron()
        self.classifier3 = ModifiedPerceptron()
        

    def fit(self, X, y):
        y1 = np.where(y == 0, 1, 0)
        self.classifier1.fit(X, y1)

        y2 = np.where(y == 1, 1, 0)

        self.classifier2.fit(X, y2)

        y3 = np.where(y == 2, 1, 0)
        self.classifier3.fit(X, y3)

    def predict(self, X):
        predictions = np.array([
            self.classifier1.predict(X),
            self.classifier2.predict(X),
            self.classifier3.predict(X)
        ]).T
        
        # if the sample was 1D, return the max along axis 0
        if len(predictions.shape) == 1:
            return np.argmax(predictions, axis=0)

        # otherwise, return the max along axis 1
        return np.argmax(predictions, axis=1)





# loading iris dataset
iris = datasets.load_iris()
X = iris.data[:]
y = iris.target

# the test book says most scikit learn algorithms use the one-versus-rest test.
# could we just train a perceptron for each subset of the features? (each one performs binary classification?)
multiple_perceptrons = MultiplePerceptrons()
multiple_perceptrons.fit(X, y)

random_sample_index = 35

print(f'Predicted: {multiple_perceptrons.predict(X)}')
print(f'Actual: {y}')
#print(multiple_perceptrons.predict(X[0:50]))


