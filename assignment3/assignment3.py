import numpy as np
import idx2numpy

from sklearn.model_selection import train_test_split
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


#### Loading the data into numpy arrays
# TODO: turn the directory paths into command line arguments?
mnist_data_dir_path = './mnist-data'
fashion_data_dir_path = './fashion-data'

X_train, y_train, X_test, y_test = load_data(fashion_data_dir_path)

#### sklearn pipeline (fashion)

fashion_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=50),
    SVC(kernel='poly', random_state=random_state),
    verbose=True
)
fashion_pipeline.fit(X_train, y_train)
print(f'Percentage of correctly identified samples: {fashion_pipeline.score(X_test, y_test)*100:.2f}')

