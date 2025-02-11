from sklearn import datasets
import numpy as np



"""
(4 points) Modify the class Perceptron given in the textbook such that the bias data field b is absorbed by
the weight vector w . Your program is required to be compatible with the training program in the textbook.

"""

class Perceptron:
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
        
        for _ in range(self.n_iter):
            errors = 0
            # for each (x[i], y[i]) in (X, y)
            for xi, target in zip(X, y):
                # compute eta * (target - prediction)
                update = self.eta * (target - self.predict(xi))
                # update weights and bias
                self.w_ += update * np.hstack((xi, np.array([1])))
                # self.b_ += update
                # calculate the error (only 0 or 1)
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        # this computes the inner product between X and w and then adds
        # b element wise.
        # add an extra 1 to to X then compute the inner product like normal
        X_modified = np.hstack((X, np.array([1])))
        return np.dot(X_modified, self.w_)
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)


"""
(6 points) A perceptron can only be used for binary classification, however, the Iris dataset has 3 classes:
setosa, versicolor and virginica. If you are only allowed to use perceptrons but the number is not limited,
how would you like to perform a multiclass classification for the Iris data set (all features)? Please write a
program (demo) for this task using the new perceptron class developed in the previous task.

"""

class MulticlassPerceptrons:

    def __init__(self):
        self.classifier1 = Perceptron()
        self.classifier2 = Perceptron()
        self.classifier3 = Perceptron()
        

    def fit(self, X, y):
        y1 = np.where(y == 0, 0, 1)
        self.classifier1.fit(X, y1)

        y2 = np.where(y == 1, 0, 1)

        self.classifier2.fit(X, y2)

        y3 = np.where(y == 2, 0, 1)
        self.classifier3.fit(X, y3)

    def predict(self, X):
        """
        
        """
        pred1 = self.classifier1.predict(X)
        pred2 = self.classifier2.predict(X)
        pred3 = self.classifier3.predict(X)

        if pred1 and not (pred2 and pred3):
            return 0
        elif pred2 and not (pred1 and pred3):
            return 1
        else:
            return 2





# loading iris dataset
iris = datasets.load_iris()
X = iris.data[:]
y = iris.target
print(f'Targets: {y}')

# instantiating and training a model
model = Perceptron(random_state=None)
model.fit(X, y)

# the test book says most scikit learn algorithms use the one-versus-rest test.
# could we just train a perceptron for each subset of the features? (each one performs binary classification?)
multiclass_perceptrons = MulticlassPerceptrons()
multiclass_perceptrons.fit(X, y)

random_sample_index = 1
random_sample_features = X[random_sample_index]
random_sample_target = y[random_sample_index]

print(f'Predicted: {multiclass_perceptrons.predict(random_sample_features)}')
print(f'Actual: {random_sample_target}')

