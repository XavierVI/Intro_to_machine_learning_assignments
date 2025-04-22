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
    def __init__(self, input_size):
        super().__init__()

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


class FnnEnsemble:
    def __init__(self, model_class, num_of_models, input_size, device):
        # create a list of models
        models = [model_class().to(device) for _ in range(num_of_models)]

        # stack their weights together
        self.param, self.buffers = stack_module_state(models)

        self.models = []

    def predict(X):
        pass

