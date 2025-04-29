import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models import *
from train import *

from move_reviews import load_movie_reviews

import time


def main():
    batch_size = 64
    dataset_size = 30_000
    max_features = 10_000
    num_of_models = 6
    # set device to be a CUDA device if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # only use the training data
    X_train, X_test, y_train, y_test \
        = load_movie_reviews('./movie_data.csv', dataset_size, max_features)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    #### create a set of bootstraps
    X_train_bootstraps = []
    y_train_bootstraps = []

    for i in range(num_of_models):
        features, labels = create_bootstrap(10_000, X_train, y_train)
        X_train_bootstraps.append(features)
        y_train_bootstraps.append(labels)


    test_dataset = TensorDataset(X_test, y_test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    #### Define the layers for each model
    models = [FNNWithDropout(X_train.shape[1]).to(device) 
                    for _ in range(num_of_models)]

    ensemble = zip(models, X_train_bootstraps, y_train_bootstraps)

    start = time.time()
    for i, (model, X_train, y_train) in enumerate(ensemble):
        train_dataset = TensorDataset(X_train, y_train)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
        test_acc = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=10,
            learning_rate=0.0001,
            l2_reg=5e-6
        )
        end = time.time()
        time_cost = end - start

        print(f'Avg. accuracy and time cost for model {i}: {test_acc*100:.2f}, {time_cost:.4f}s')

    #### evaluating the ensemble
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    predictions = [nn.functional.softmax(model(X_test), dim=1) for model in models]
    predictions = torch.stack(predictions)
    predictions = torch.mean(predictions, dim=0)
    predicted = torch.argmax(predictions, dim=1)
    correct = (predicted == y_test).sum().item()
    total = y_test.size(0)
    accuracy = 100.0 * correct / total
    print(f'Ensemble accuracy: {accuracy:.2f}%')



if __name__ == '__main__':
    main()
