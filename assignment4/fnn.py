import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset

import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords



class FNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        # hidden layers
        self.layers = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        

    def forward(self, x):
        return self.layers(x)


def load_movie_reviews(csv_file, dataset_size, max_features):
    """
    returns the data as two tensors
    """
    porter = PorterStemmer()


    def tokenizer_porter(text):
        return [porter.stem(word) for word in text.split()]
    
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
    labels = torch.tensor(df['sentiment'].values)
    features = tfidf_vectorizer.fit_transform(df['review'])
    features = torch.tensor(features.toarray(), dtype=torch.float32)
    return features, labels


def train_model(
    model,
    train_loader,
    test_loader,
    device,
    num_epochs,
    learning_rate,
    l2_reg
    ):
    #### instantiate optimizer and loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=l2_reg # L2 regularization
    )

    # Training loop
    for epoch in range(num_epochs):
        # accumulators
        avg_loss = 0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            # forward pass
            predictions = model(X)
            # compute the loss
            loss = loss_fn(predictions, y)
            avg_loss += loss.item()

            # backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # compute the accuracy
            _, y_hat = torch.max(predictions.data, dim=1)
            correct += (y_hat == y).sum().item()
            total += y.size(0)
            # print(f'Batch Loss: {loss}')
            # print(f'Avg. Accuracy: {correct * 100 / total}')

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print('============================================================================')
        print(f'Avg. Loss: {avg_loss/len(train_loader):.4f}')
        print(f'Avg. Accuracy: {correct*100/total:.4f}')
        test_accuracy = get_test_accuracy(model, test_loader, device)
        print(f'Test accuracy: {test_accuracy*100:.4f}')
    
    print('Training finished!')


def get_test_accuracy(model, test_loader, device): 
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            predictions = model(X)
            # compute the accuracy
            _, y_hat = torch.max(predictions.data, dim=1)
            correct += (y_hat == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    batch_size = 64
    dataset_size = 50_000
    max_features =  20_000

    #### set device to be a CUDA device if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    

    X, y = load_movie_reviews('./movie_data.csv', dataset_size, max_features)
    dataset = TensorDataset(X, y)

    # Create train/test indices
    train_indices, test_indices = train_test_split(
        range(X.size(0)),
        test_size=0.3,
        random_state=1,
        shuffle=True
    )

    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
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

    #### instantiate the model and load it on the device
    model = FNN(X.size(1))
    model.to(device)
            
    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=50,
        learning_rate=0.0001,
        l2_reg=1e-5
    )



if __name__ == '__main__':
    main()
