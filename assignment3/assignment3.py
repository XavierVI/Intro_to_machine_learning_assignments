import numpy as np
import idx2numpy

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC

import os



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


def grid_search(pipeline, params, X_train, y_train):
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring='accuracy',
        cv=2,
        n_jobs=2,
        # the model will return with the best parameters
        # if this value is True
        refit=True
    )

    gs.fit(X_train, y_train)

    # return the pipeline with the best parameters
    return gs.best_estimator_

def find_best_hyper_params(pipeline, X_train, y_train, params):
    # dictionaries for grid search
    # TODO: add values for each parameter (at least 8 for each)

    pass


def main():
    random_state = 1
   
    #### Dataset directory paths
    mnist_data_dir_path = './mnist-data'
    fashion_data_dir_path = './fashion-data'
   
    #### Grid search parameters and PCA components
    params_linear = {
        'svc__kernel': ['linear'],
        'svc__C': [1, 2, 4, 8, 12, 16, 32, 64]
    }

    params_rbf = {
        'svc__kernel': ['rbf'],
        'svc__C': [1, 2, 4, 8, 12, 16, 32, 64],
        'svc__gamma': [1, 2, 4, 8, 16, 32, 64],
    }

    params_poly = {
        'svc__kernel': ['poly'],
        'svc__C': [1, 2, 4, 8, 12, 16, 32, 64],
        'svc__gamma': [1, 2, 4, 8, 16, 32, 64],
        'svc__degree': [2, 3, 4]
    }
   
    PCA_comps = [50, 100, 200]

    #### Loading data
    X_train, y_train, X_test, y_test = load_data(mnist_data_dir_path)


    for n_comps in PCA_comps:
        # Creating our pipeline
        pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_comps),
            SVC(random_state=random_state),
            verbose=False
        )
        print(f'Finding best hyperparameters for MNIST, n_comps = {n_comps}')
        
        #### Get the best model with a linear kernel
        linear_model = grid_search(
            pipeline, params_linear, X_train[:1000], y_train[:1000])
        linear_accuracy = linear_model.score(X_test, y_test)
        linear_cm = confusion_matrix(y_test, linear_model.predict(X_test))
        print(linear_cm)


        



if __name__ == '__main__':
    main()