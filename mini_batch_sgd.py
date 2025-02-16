import numpy as np
from sklearn.datasets import load_iris
from textbook_code import AdalineGD
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""
Task 4:
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
        #########################################
        # compute the number of subsets
        num_of_subsets = X.shape[0] // batch_size
        #########################################

        for _ in range(self.n_iter):
            # shuffle the dataset
            if self.shuffle:
                X, y = self._shuffle(X, y)

            losses = []
            #######################################
            # for each subset of the data
            for i in range(num_of_subsets):
                # compute the indices of the first and last
                # instance in the subset
                start = batch_size * i
                end = start + batch_size
                # pass the subset to _update_weights
                losses.append(self._update_weights(X[start:end], y[start:end]))
            #####################################
            
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
        ############################################
        # modified update rule to work with batches
        if len(xi.shape) == 1:
            self.w_ += self.eta * 2.0 * xi * error
        else:
            self.w_ += self.eta * 2.0 * xi.T @ error
        ###########################################
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
# set the labels to be 1 for setosa and zero for the rests
y = np.where(y == 0, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

epochs = 10
learning_rate = 0.01
random_state_value = 1

adaline_mini_batch_SGD = ModifiedAdalineSGD(
    eta=learning_rate,
    n_iter=epochs,
    random_state=random_state_value
)
adaline_SGD = ModifiedAdalineSGD(
    eta=learning_rate,
    n_iter=epochs,
    random_state=random_state_value
) 
adaline_GD = AdalineGD(
    eta=learning_rate,
    n_iter=epochs,
    random_state=random_state_value
)

# train each model
adaline_GD.fit(X, y)
adaline_mini_batch_SGD.fit_mini_batch_SGD(X_train, y_train, batch_size=2)
adaline_SGD.fit(X, y)


# test each model (number of misclassifications)
adaline_GD_performance = np.sum(adaline_GD.predict(X_test) == y_test)
adaline_SGD_performance = np.sum(adaline_SGD.predict(X_test) == y_test)
adaline_mini_batch_SGD_performance = np.sum(adaline_mini_batch_SGD.predict(X_test) == y_test)

misclassifications = [
    adaline_GD_performance,
    adaline_SGD_performance,
    adaline_mini_batch_SGD_performance
]
models = ['Adaline GD', 'Adaline SGD', 'Adaline Mini-batch SGD']
bar_colors = ['tab:blue', 'tab:red', 'tab:orange']

plt.bar(models, misclassifications, color=bar_colors)
plt.xlabel('Model')
plt.ylabel('Number of Misclassifications')
plt.title('Bar chart comparing the number of misclassifications')
# plt.show()
plt.savefig('task4_bar_chart')



