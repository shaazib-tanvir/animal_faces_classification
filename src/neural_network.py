from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._layers = nn.Sequential(
            nn.Conv2d(3, 3, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 8),
            nn.Flatten(),
            nn.Linear(12288, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3),
            nn.Softmax(1)
        )

    
    def forward(self, x):
        return self._layers(x)
