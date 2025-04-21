import torch.nn as nn

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
