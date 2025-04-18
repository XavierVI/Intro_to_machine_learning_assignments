import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU()
        )
        

    def forward(self, x):
        return self.layers(x)


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
    
    #### instantiate the model and load it on the device
    model = FNN()
    model.to(device)
    


if __name__ == '__main__':
    main()
