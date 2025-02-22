from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split

"""
Task 2:
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
        ###################################################
        # add the bias to the end of the weight matrix
        self.w_ = np.hstack((self.w_, np.array([0])))
        ##################################################

        ###################################################
        # removed the bias and add an extra computation
        # self.b_ = np.float64(0.)
        ##################################################

        # a list to accumulate the errors
        self.errors_ = []

        # print(f'Started training with X: {X.shape}')
        
        for _ in range(self.n_iter):
            errors = 0
            # for each (x[i], y[i]) in (X, y)
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                ###############################################
                # update weights
                self.w_ += update * np.concatenate((xi, [1]))
                ##############################################
                
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        #############################################
        # if X is a 1D array, add a 1 to the end
        if len(X.shape) == 1:
            X_modified = np.concatenate((X, [1]))
        else:
            # otherwise, add a ones to the end of each row
            X_modified = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ############################################
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
        # create three instances of a classifier
        self.setosa_classifier = ModifiedPerceptron()
        self.veriscolor_classifier = ModifiedPerceptron()
        self.verginica_classifier = ModifiedPerceptron()
        

    def fit(self, X, y):
        setosa_labels = np.where(y == 0, 1, 0)
        self.setosa_classifier.fit(X, setosa_labels)

        versicolor_labels = np.where(y == 1, 1, 0)
        self.veriscolor_classifier.fit(X, versicolor_labels)

        verginica_labels = np.where(y == 2, 1, 0)
        self.verginica_classifier.fit(X, verginica_labels)

    def predict(self, X):
        predictions = np.array([
            self.setosa_classifier.predict(X),
            self.veriscolor_classifier.predict(X),
            self.verginica_classifier.predict(X)
        ])
        
        return np.argmax(predictions, axis=0)



iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# the test book says most scikit learn algorithms use the one-versus-rest test.
# could we just train a perceptron for each subset of the features? (each one performs binary classification?)
multiple_perceptrons = MultiplePerceptrons()
multiple_perceptrons.fit(X_train, y_train)

predictions = multiple_perceptrons.predict(X_test)

print(f'Predicted labels: {predictions}')
print(f'Actual labels: {y_test}')
print(f'Number of misclassifications: {np.sum(predictions != y_test)} / {y_test.shape[0]}')


