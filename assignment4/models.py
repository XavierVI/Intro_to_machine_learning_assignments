import torch
import torch.nn as nn
from torch.func import stack_module_state


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


class FNNWithDropout(nn.Module):
    def __init__(self, input_size, layers=None):
        super().__init__()

        if layers is not None:
            self.layers = layers
            return

        # hidden layers
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            
            nn.Linear(256, 64),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.Dropout(0.4),
            nn.ReLU(),
            
            nn.Linear(32, 2)
        )


    def forward(self, x):
        return self.layers(x)




