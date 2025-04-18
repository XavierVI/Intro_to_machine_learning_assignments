import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import pandas as pd


from sklearn.feature_extraction.text import TfidfVectorizer



class MovieReviewsDataset(Dataset):

    def __init__(self, csv_file_path, dataset_size, transform=None):
        # read the csv file path
        df = pd.read_csv(csv_file_path)
        # shrink the dataset
        df = df[:dataset_size]
        self.labels = torch.tensor(df['sentiment'].values)

        # creating the feature vector
        tfidf_vectorizer = TfidfVectorizer(
            tokenizer=self.tokenizer,
            # max_features=10_000
            stop_words='english'
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
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
        

    def forward(self, x):
        return torch.argmax(self.layers(x))



def train_model(model: nn.Module, dataloader, device):
    num_epochs = 10

    #### instantiate optimizer
    learning_rate = 0.01
    batch_size = 32
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for X, y in dataloader:
            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    print('Training finished!')

    # Example of making predictions
    with torch.no_grad():
        new_input = torch.randn(5, input_size)
        predictions = torch.argmax(model(new_input), dim=1)
        print('Predictions for new inputs:', predictions)
    


def main():
    #### set device to be a CUDA device if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    

    dataset = MovieReviewsDataset('./movie_data.csv', 10)

    #### instantiate the model and load it on the device
    model = FNN(dataset.num_of_features())
    model.to(device)

    X, y = dataset[:1]
    X = X.to(device)
    print(model(X))
    


if __name__ == '__main__':
    main()
