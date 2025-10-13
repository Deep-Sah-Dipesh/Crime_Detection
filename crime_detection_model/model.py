import torch.nn as nn
import config

class CrimeClassifier(nn.Module):
    """
    Defines the neural network architecture for the crime classifier.
    """
    def __init__(self):
        super(CrimeClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(config.INPUT_FEATURE_SIZE, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5), # Regularization to prevent overfitting
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, config.NUM_CLASSES)
        )

    def forward(self, x):
        return self.network(x)

