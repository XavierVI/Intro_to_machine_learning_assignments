import numpy as np
import idx2numpy

import os

#### Loading the data into numpy arrays
# TODO: turn the directory paths into command line arguments?
mnist_data_dir_path = './mnist-data'
fashion_data_dir_path = './fashion-data'

# loading mnist data
mnist_training_images = idx2numpy.convert_from_file(os.path.join(mnist_data_dir_path, 'train-images-idx3-ubyte'))
mnist_training_labels = idx2numpy.convert_from_file(os.path.join(mnist_data_dir_path, 'train-labels-idx1-ubyte'))
mnist_testing_images = idx2numpy.convert_from_file(os.path.join(mnist_data_dir_path, 't10k-images-idx3-ubyte'))
mnist_testing_labels = idx2numpy.convert_from_file(os.path.join(mnist_data_dir_path, 't10k-labels-idx1-ubyte'))


# loading fashion mnist data
fashion_training_images = idx2numpy.convert_from_file(os.path.join(fashion_data_dir_path, 'train-images-idx3-ubyte'))
fashion_training_labels = idx2numpy.convert_from_file(os.path.join(fashion_data_dir_path, 'train-labels-idx1-ubyte'))
fashion_testing_images = idx2numpy.convert_from_file(os.path.join(fashion_data_dir_path, 't10k-images-idx3-ubyte'))
fashion_testing_labels = idx2numpy.convert_from_file(os.path.join(fashion_data_dir_path, 't10k-labels-idx1-ubyte'))

