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
            stop_words='english',
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
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 2),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
        

    def forward(self, x):
        return self.layers(x)



def train_model(model: nn.Module, train_loader, device):
    # splitting the dataset
    num_epochs = 50

    #### instantiate optimizer
    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            pred = model(X)
            # compute the loss
            loss = criterion(pred, y)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    print('Training finished!')


def main():
    batch_size = 32
    dataset_size = 16_384
    max_features =  20_000

    #### set device to be a CUDA device if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    

    dataset = MovieReviewsDataset('./movie_data.csv', dataset_size, max_features)

    # Create train/test indices
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.3,
        random_state=42,
        shuffle=True
    )
    train_dataset = Subset(dataset, train_indices)
    print(f'Training dataset has: {len(train_dataset)} samples')
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    

    #### instantiate the model and load it on the device
    model = FNN(dataset.num_of_features())
    model.to(device)
    
    train_model(model, train_loader, device)
    


if __name__ == '__main__':
    main()
