import torch
from torch.utils.data import TensorDataset, DataLoader


from models import *
from train import *

from move_reviews import load_movie_reviews

import time


def main():
    batch_size = 64
    dataset_size = 50_000
    max_features = 20_000

    # set device to be a CUDA device if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

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

    # instantiate the model and load it on the device
    model = FNN(X_train.size(1))
    model.to(device)

    start = time.time()
    test_acc = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=10,
        learning_rate=0.0001,
        l2_reg=0.0
    )
    end = time.time()
    time_cost = end - start
    print(f'Test accuracy and time cost: {test_acc*100:.2f}, {time_cost:.4f}s')


if __name__ == '__main__':
    main()
