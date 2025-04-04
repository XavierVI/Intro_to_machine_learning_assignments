import numpy as np
import idx2numpy

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC

import os

random_state = 1

def load_data(dir_path: str):
    """
    
    dir_path: the path to the directory with the data
    """
    num_of_training_samples = 60_000
    num_of_testing_samples = 10_000

    # loading training data
    training_images = idx2numpy.convert_from_file(
        os.path.join(dir_path, 'train-images-idx3-ubyte'))
    training_labels = idx2numpy.convert_from_file(
        os.path.join(dir_path, 'train-labels-idx1-ubyte'))

    # reshape training data
    training_images = training_images.reshape((num_of_training_samples, -1))

    # loading testing data
    testing_images = idx2numpy.convert_from_file(
        os.path.join(dir_path, 't10k-images-idx3-ubyte'))
    testing_labels = idx2numpy.convert_from_file(
        os.path.join(dir_path, 't10k-labels-idx1-ubyte'))

    # reshape testing data
    testing_images = testing_images.reshape((num_of_testing_samples, -1))

    return training_images, training_labels, testing_images, testing_labels


def grid_search(params, n_components, X_train, y_train):
    # sklearn pipeline (fashion)
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components),
        SVC(random_state=random_state),
        verbose=False
    )

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring='accuracy',
        cv=2,
        n_jobs=2
    )

    gs.fit(X_train, y_train)

    return gs.best_params_, gs.best_score_

#### Loading the data into numpy arrays
# TODO: turn the directory paths into command line arguments?
mnist_data_dir_path = './mnist-data'
fashion_data_dir_path = './fashion-data'

X_train, y_train, X_test, y_test = load_data(fashion_data_dir_path)

#### dictionary for grid search
grid_search_params_linear = {
    'svc__kernel': ['linear'],
    'svc__C': [1, 10]
}

grid_search_params_rbf = {
        'svc__kernel': ['rbf'],
        'svc__C': [1, 10],
        'svc__gamma': [1]
}

grid_search_params_poly = {
    'svc__kernel': ['poly'],
    'svc__C': [1, 10],
    'svc__gamma': [1],
    'svc__degree': [2]
}

PCA_values = [50, 100, 200]

for pca_val in PCA_values:
    print(grid_search(grid_search_params_linear, pca_val, X_train[:1000], y_train[:1000]))
    print(grid_search(grid_search_params_rbf, pca_val, X_train[:1000], y_train[:1000]))
    print(grid_search(grid_search_params_poly, pca_val, X_train[:1000], y_train[:1000]))
