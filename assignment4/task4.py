import torch
from torch.utils.data import TensorDataset, DataLoader


from models import *
from train import *

from movie_reviews import load_movie_reviews

import time

def main():
    batch_size = 64
    dataset_size = 30_000
    max_features = 10_000

    # set device to be a CUDA device if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # only use the training data
    X_train, _, y_train, _ \
        = load_movie_reviews('./movie_data.csv', dataset_size, max_features)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train)

    train_dataset = TensorDataset(X_train, y_train)

    start = time.time()
    _, avg_acc = perform_kfold(
        model_class=FNN,
        dataset=train_dataset,
        input_size=X_train.size(1),
        batch_size=batch_size,
        k_folds=5,
        num_epochs=20,
        learning_rate=0.0001,
        l2_reg=5e-6
    )
    end = time.time()
    time_cost = end - start
    
    print(f'Avg. accuracy and time cost: {avg_acc:.2f}, {time_cost:.4f}s')


if __name__ == '__main__':
    main()
