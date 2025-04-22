import torch
from torch.utils.data import TensorDataset, Subset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from models import *
from train import *

import pandas as pd


def load_movie_reviews(csv_file, dataset_size, max_features):
    """
    returns the data as two tensors
    """
    # read the csv file path
    df = pd.read_csv(csv_file)
    # shrink the dataset
    df = df[:dataset_size]

    # creating the feature vector
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=lambda text: text.split(),
        # tokenizer=tokenizer_porter,
        stop_words='english',
        max_features=max_features,
        # ngram_range=(1, 2)
    )
    # labels = torch.tensor(df['sentiment'].values)
    X = tfidf_vectorizer.fit_transform(df['review']).toarray()
    y = df['sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=1,
        shuffle=True
    )

    return X_train, X_test, y_train, y_test

def main():
    batch_size = 64
    dataset_size = 40_000
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
    model = FNNWithDropout(X_train.size(1))
    model.to(device)

    # test_acc, time_cost = train_model(
    #     model=model,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     device=device,
    #     num_epochs=25,
    #     learning_rate=0.0001,
    #     l2_reg=5e-6
    # )
    # print(f'Test accuracy and time cost: {test_acc*100}, {time_cost:.4f}s')

    perform_kfold(
        model_class=FNN,
        dataset=train_dataset,
        input_size=X_train.shape[1],
        num_epochs=10,
        learning_rate=0.0001,
        l2_reg=5e-6
    )

if __name__ == '__main__':
    main()