import numpy as np
import idx2numpy

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC

import os

import matplotlib.pyplot as plt
import seaborn as sns

import sys

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
    if len(sys.argv) >= 2:
        data_dir_path = sys.argv[1]
        
    else:
        print("Please specify a path to the data using 'python assignment3.py /path/to/data")
        print("or use ./path/to/data")
        return 1

    #### Dictionary for storing table data
    table = {
        'PCA Comps': [],
        'Kernel': [],
        'Accuracy': []
    }
   
    #### Grid search parameters and PCA components
    params = [
        {
            'svc__kernel': ['linear'],
            'svc__C': [1, 2, 4, 8, 12, 16, 32, 64]
        },
        {
            'svc__kernel': ['rbf'],
            'svc__C': [1, 2, 4, 8, 12, 16, 32, 64],
            'svc__gamma': [1, 2, 4, 8, 16, 32, 64],
        },
        {
            'svc__kernel': ['poly'],
            'svc__C': [1, 2, 4, 8, 12, 16, 32, 64],
            'svc__gamma': [1, 2, 4, 8, 16, 32, 64],
            'svc__degree': [2, 3, 4]
        }
    ]
   
    PCA_comps = [50, 100, 200]

    #### Loading data
    X_train, y_train, X_test, y_test = load_data(data_dir_path)
    class_names = np.unique(y_test)


    for n_comps in PCA_comps:
        # Creating our pipeline
        pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_comps),
            SVC(random_state=random_state),
            verbose=False
        )
        print(f'Finding best hyperparameters for MNIST, n_comps = {n_comps}')

        # 1 row, 3 columns of subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 12))

        for i, param_list in enumerate(params):
            #### Get the best model with parameters in param_list
            model = grid_search(
                pipeline, param_list, X_train[:1000], y_train[:1000])
            accuracy = model.score(X_test, y_test)
            cm = confusion_matrix(y_test, model.predict(X_test))

            #### Adding values to the table
            table['PCA Comps'].append(n_comps)
            table['Kernel'].append(param_list['svc__kernel'][0])
            table['Accuracy'].append(accuracy)
            
            
            #### creating an image of a confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=class_names,
                        ax=axes[i])
            
            axes[i].set_title(f'{param_list["svc__kernel"][0]} Kernel')
            axes[i].set_ylabel('True label')
            axes[i].set_xlabel('Predicted label')
            
        fig.suptitle(f'MNIST, PCA={n_comps}')
        # plt.show()
        plt.savefig(fname=f'./images/cm_pca_{n_comps}_{param_list['svc__kernel'][0]}_kernel')

    #### Plotting the table
    for n_comps, kernel, acc in zip(table['PCA Comps'], table['Kernel'], table['Accuracy']):
        print(f"{n_comps} & {kernel} & {acc:.2f}  \\\\")


if __name__ == '__main__':
    main()