import numpy as np

class LinearSCV:

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):

        # Initialize weights and bias
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        
        self.losses_ = []

        for i in range(self.n_iter):
            loss = 0
            for xi, target in zip(X, y):
                # Compute hinge loss gradient
                margin = target * self.net_input(xi)
                if margin >= 1:

                    # L2-Regularization
                    self.w_ -= self.eta * (2 * self.lambdaNum * self.w_)
                else:
                    # Misclassified, update weights and bias
                    self.w_ -= self.eta * (2 * self.lambdaNum * self.w_ - target * xi)
                    self.b_ += self.eta * target

                # Compute hinge loss function 
                loss += max(0, 1 - margin)

            # Store the average hinge loss
            self.losses_.append(loss / len(y))

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0, 1, -1)