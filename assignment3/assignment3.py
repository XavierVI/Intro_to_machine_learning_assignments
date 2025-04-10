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
    print(f'Loading data stored in {dir_path }')
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
        n_jobs=10,
        # the model will return with the best parameters
        # if this value is True
        refit=True
    )

    gs.fit(X_train, y_train)

    # return the pipeline with the best parameters
    return (gs.best_estimator_, gs.refit_time_, gs.best_params_)


def run_task(data_dir_path, dataset_name):
    """Completes task 4 using mnist dataset"""
    random_state = 1
    subset = 10_000

    # Dictionary for storing table data
    table = {
        'PCA Comps': [],
        'Kernel': [],
        'Accuracy': [],
        'Time Cost': []
    }

    params_table = {
        'PCA Comps': [],
        'kernel': [],
        'C': [],
        'gamma': [],
        'degree': []
    }

    # Grid search parameters and PCA components
    param_range = [1e-4, 1e-2, 1, 10, 100, 1000]
    params = [
        {
            'svc__kernel': ['linear'],
            'svc__C': param_range
        },
        {
            'svc__kernel': ['rbf'],
            'svc__C': param_range,
            'svc__gamma': param_range,
        },
        {
            'svc__kernel': ['poly'],
            'svc__C': param_range,
            'svc__gamma': param_range,
            'svc__degree': [2, 3, 4, 5]
        }
    ]

    PCA_comps = [50, 100, 200]

    # Loading data
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
        print(f'Finding best hyperparameters for {dataset_name} with n_comps = {n_comps}')

        # 1 row, 3 columns of subplots
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        for i, param_list in enumerate(params):
            # Get the best model with parameters in param_list
            model, time_cost, best_params = grid_search(
                pipeline, param_list, X_train[:subset], y_train[:subset])
            # print(f'PCA Comps = {n_comps}, Best parameters: {best_params}')
            accuracy = model.score(X_test, y_test)
            cm = confusion_matrix(y_test, model.predict(X_test))

            # Adding values to the table
            table['PCA Comps'].append(n_comps)
            table['Kernel'].append(param_list['svc__kernel'][0])
            table['Accuracy'].append(accuracy)
            table['Time Cost'].append(time_cost)

            if param_list['svc__kernel'][0] == 'linear':
                params_table['PCA Comps'].append(n_comps)
                params_table['kernel'].append(param_list['svc__kernel'][0])
                params_table['C'].append(best_params['svc__C'])
                params_table['gamma'].append(None)
                params_table['degree'].append(None)
            elif param_list['svc__kernel'][0] == 'rbf':
                params_table['PCA Comps'].append(n_comps)
                params_table['kernel'].append(param_list['svc__kernel'][0])
                params_table['C'].append(best_params['svc__C'])
                params_table['gamma'].append(best_params['svc__gamma'])
                params_table['degree'].append(None)
            else:
                params_table['PCA Comps'].append(n_comps)
                params_table['kernel'].append(param_list['svc__kernel'][0])
                params_table['C'].append(best_params['svc__C'])
                params_table['gamma'].append(best_params['svc__gamma'])
                params_table['degree'].append(best_params['svc__degree'])


            # creating an image of a confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=class_names,
                        ax=axes[i])

            axes[i].set_title(f'{param_list["svc__kernel"][0]} Kernel')
            axes[i].set_ylabel('True label')
            axes[i].set_xlabel('Predicted label')

        fig.suptitle(f'Confusion Matrices {dataset_name} PCA={n_comps}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()
        plt.savefig(
            fname=f'./images/cm_{dataset_name}_pca_{n_comps}_{param_list['svc__kernel'][0]}_kernel')

    print('============================')
    print('Parameter table')
    print('============================')
    # printing the table
    for n_comps, kernel, C, gamma, degree in zip(
        params_table['PCA Comps'], params_table['kernel'],
        params_table['C'], params_table['gamma'], params_table['degree']):
        print(f"{n_comps} & {kernel} & {C} & {gamma} & {degree} \\\\")

    print('============================')
    print('Performance table')
    print('============================')
    # printing the table
    for n_comps, kernel, acc, time_cost in zip(table['PCA Comps'], table['Kernel'], table['Accuracy'], table['Time Cost']):
        print(f"{n_comps} & {kernel} & {acc:.3f} & {time_cost:.3f} \\\\")



def main():
    mnist_data_path = './mnist-data/'
    fashion_data_path = './fashion-data/'

    if not os.path.exists('./images'):
        os.mkdir('images')
    
    if len(sys.argv) < 2:
        #### Run task for both datasets
        run_task(mnist_data_path, 'MNIST')
        run_task(fashion_data_path, 'Fashion_MNIST')
    elif sys.argv[1] == 'mnist':
        run_task(mnist_data_path, 'MNIST')
    elif sys.argv[1] == 'fashion':
        run_task(fashion_data_path, 'Fashion_MNIST')


if __name__ == '__main__':
    main()