import torch
from torch.utils.data import TensorDataset, DataLoader


from models import *
from train import *

from move_reviews import load_movie_reviews

import time



def main():
    batch_size = 64
    dataset_size = 40_000
    max_features = 20_000
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

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    #### Define the layers for each model
    model1 = FNNWithDropout(X_train.size(1), layers=nn.Sequential(
        nn.Linear(X_train.size(1), 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2)
    ))

    model2 = FNNWithDropout(X_train.size(1), layers=nn.Sequential(
        nn.Linear(X_train.size(1), 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2)
    ))

    model3 = FNNWithDropout(X_train.size(1), layers=nn.Sequential(
        nn.Linear(X_train.size(1), 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2)
    ))

    model4 = FNNWithDropout(X_train.size(1), layers=nn.Sequential(
        nn.Linear(X_train.size(1), 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2)
    ))

    model5 = FNNWithDropout(X_train.size(1), layers=nn.Sequential(
        nn.Linear(X_train.size(1), 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2)
    ))

    models = [model1, model2, model3, model4, model5]

    start = time.time()
    for i, model in enumerate(models):
        model.to(device)
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

        print(f'Avg. accuracy and time cost for model {i}: {test_acc:.2f}, {time_cost:.4f}s')


if __name__ == '__main__':
    main()
