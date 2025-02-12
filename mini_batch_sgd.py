import numpy as np
from sklearn.datasets import load_iris

"""
(4 points) In the textbook, both SGD and mini-batch GD are explained. Let us write a new training function
fit mini batch SGD in the class AdalineSGD such that the new function combines SGD and mini-batch
GD in training. In every epoch, the mini-batch SGD method updates the learning parameters based on a
randomly selected subset of the training data. The size (number of samples) of the subset is called the batch
size which is a hyperparameter decided by the user.
"""

class ModifiedAdalineSGD:
    """ADAptive LInear NEuron classifier.
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent 
        cycles.
    random_state : int
        Random number generator seed for random weight 
        initialization.
    
    
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    losses_ : list
        Mean squared error loss function value averaged over all
        training examples in each epoch.
    
    
    """
    def __init__(self, eta=0.01, n_iter=10,
                 shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
    
    def fit(self, X, y):
        """ Fit training data.
        
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
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses) 
            self.losses_.append(avg_loss)
        return self

    def fit_mini_batch_SGD(self, X, y, batch_size=4):
        # use numpy's vector operations to update the hyperparameters
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        num_of_subsets = X.shape[0] // batch_size

        for _ in range(self.n_iter):
            # shuffle the dataset
            if self.shuffle:
                X, y = self._shuffle(X, y)

            losses = []
            # for each subset of the data
            for i in range(num_of_subsets):
                start = batch_size * i
                end = start + batch_size
                losses.append(self._update_weights(X[start:end], y[start:end]))
            
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)

        return self
    
    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self
    
    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01,
                                   size=m)
        self.b_ = np.float_(0.)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        # modified update rule to work with batches
        self.w_ += self.eta * 2.0 * xi.T @ (error)
        self.b_ += self.eta * 2.0 * error.sum()
        loss = error**2
        return loss
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        """Compute linear activation"""
        return X
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X))
                        >= 0.5, 1, 0)


# loading iris dataset
iris = load_iris()
X = iris.data[:]
y = iris.target

model = ModifiedAdalineSGD(random_state=None)
model.fit_mini_batch_SGD(X, y, batch_size=2)

print(f'Predicted: {model.predict(X[42])}')
print(f'Actual: {y[42]}')


