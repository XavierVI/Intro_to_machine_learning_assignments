from ctypes import c_int16
import numpy as np
from urllib.error import HTTPError
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

df = pd.read_csv("./data/data_d10_n500_u2.csv")
df = df.apply(pd.to_numeric, errors='coerce')
X = df.iloc[:, :-1].values
y = df.iloc[:,-1].values
print(X.shape)

class LinearSVC:

    def __init__(self, eta=0.01, n_iter=90, random_state=0, lambdaNum=.1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.lambdaNum = lambdaNum

    def fit(self, X, y):

        # Initialize weights and bias
        print(X.shape)
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = float(0.)

        for i in range(self.n_iter):
            loss = 0
            for idx, x_i in enumerate(X): #xi, target in zip(X, y):
                #print(x_i)
                margin = (((y[idx]) * (np.dot(x_i, self.w_) - int(self.b_))) >= 1)
                if margin:
                    self.w_ -= self.eta * (2 * self.lambdaNum * self.w_)
                else:
                    self.w_ -= self.eta * (2 * self.lambdaNum * self.w_ - np.dot(x_i, y[idx]))
                    self.b_ -= self.eta * y[idx]
        print(X.shape)

    def predict(self, X):
        #print(X.shape)
        approx = np.dot(X, self.w_) - self.b_
        return np.sign(approx)

def plot_decision_regions(X, y, classifier, resolution=0.01):

    # setup marker generator and color map

    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    num_samples = xx1.ravel().shape[0]
    additional_features = np.zeros((num_samples, 8))  # Add 8 more zeroed features
    X_mesh = np.hstack((np.array([xx1.ravel(), xx2.ravel()]).T, additional_features))
    lab = classifier.predict(X_mesh)  # Now has shape (N,10)

    #lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    lab = lab.reshape(xx1.shape)

    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')


'''
                # Compute hinge loss gradient
                margin = target * self.net_input(xi)
                if margin >= 1:

                    # L2-Regularization
                    
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
        
        
    '''

sVM = LinearSVC(eta=0.01, n_iter=50,lambdaNum=.00001, random_state=1)
sVM.fit(X, y)
plot_decision_regions(X, y, sVM)

plt.title(f'LinearSVC - Classification - n_500 d_10 lambdaNum:{sVM.lambdaNum}.00001 eta:{sVM.eta}')
plt.xlabel('')
plt.ylabel('')
plt.legend(loc='upper left')
plt.show()





