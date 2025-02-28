import numpy as np


class LinearSCV:

    def __init__(self, eta=0.01, n_iter=50, C=1, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.C = C

    def fit(self, X, y):
        """
        This implementation uses GD to update the weights
        of the model.

        C: the regularization parameter
        """
        # Initialize weights and bias
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        # don't use the bias since our data generator only generates
        # hyperplanes where b = 0
        # self.b_ = np.float_(0.)
        n = X.shape[0]

        # a list to store accumulated losses
        self.losses_ = []

        for epoch in range(self.n_iter):
            sample_loss_mean = 0
            # gradient of sample loss mean
            grad_sample_loss_mean = np.zeros(self.w_.shape)

            # this for loop accumulates the hinge loss for each
            # sample in X
            for xi, target in zip(X, y):
                # 
                y_hat = self.decision_function(xi)
                # Compute the hinge loss of a single sample
                sample_loss = max(0, 1 - target * y_hat)
                sample_loss_mean += sample_loss

                # computing the gradient of the sample loss
                if sample_loss > 0:
                    grad_sample_loss_mean += -target * xi

                # otherwise the gradient of the sample loss is just 0
                # so we do nothing

            # compute the mean of the sample loss and apply regularization term
            sample_loss_mean = (self.C / n) * sample_loss_mean
            # compute the loss with L2 regularization term
            loss = sample_loss_mean + 0.5 * np.sum(self.w_**2)
            self.losses_.append(loss)

            # compute the gradient of the loss with L2 regularization
            grad_loss = (self.C / n) * grad_sample_loss_mean + self.w_
            # adjust weights
            self.w_ -= self.eta * grad_loss

    def decision_function(self, X):
        """Calculate """
        return np.dot(X, self.w_)

    def predict(self, X):
        """
        This function essentially computes the predicted class label
        by returning the sign of output from the decision function.
        """
        return np.where(self.decision_function(X) >= 0, 1, -1)
