import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

import pandas as pd


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class MovieReviewsDataset(Dataset):

    def __init__(self, csv_file_path, dataset_size, max_features, transform=None):
        # read the csv file path
        df = pd.read_csv(csv_file_path)
        # shrink the dataset
        df = df[:dataset_size]
        self.labels = torch.tensor(df['sentiment'].values)

        # creating the feature vector
        tfidf_vectorizer = TfidfVectorizer(
            tokenizer=self.tokenizer,
            # max_features=10_000
            # stop_words='english',
            max_features=max_features
        )
        features = tfidf_vectorizer.fit_transform(df['review'])
        self.features = torch.tensor(features.toarray(), dtype=torch.float32)
        self.transform = transform

    def num_of_features(self):
        return self.features.shape[1]

    def tokenizer(self, text):
        return text.split()

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        X = self.features[idx]
        y = self.labels[idx]

        if self.transform:
            X = self.transform(X)

        return X, y



class FNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        # hidden layers
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        

    def forward(self, x):
        return self.layers(x)


def train_model(model: nn.Module, train_loader, device):
    # splitting the dataset
    num_epochs = 10

    #### instantiate optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5 # L2 regularization
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
    
    print('Training finished!')


def main():
    batch_size = 32
    dataset_size = 20_000
    max_features =  10_000

    #### set device to be a CUDA device if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    

    dataset = MovieReviewsDataset('./movie_data.csv', dataset_size, max_features)

    # Create train/test indices
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.3,
        random_state=1,
        shuffle=True
    )

    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # getting a random sample
    X, y = train_dataset[0]
    print(X)
    print(y)
    print(type(y))

    

    #### instantiate the model and load it on the device
    model = FNN(dataset.num_of_features())
    model.to(device)
    
    train_model(model, train_loader, device)
    


if __name__ == '__main__':
    main()
