import torch
import torch.nn as nn
from .config import Config

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        in_channels = 1 if Config.DATASET == 'MNIST' else 3
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Output spatial size is 7x7 for MNIST (28->14->7), 8x8 for CIFAR10 (32->16->8)
        flattened_size = 32 * 7 * 7 if Config.DATASET == 'MNIST' else 32 * 8 * 8
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_model():
    if Config.MODEL_TYPE == 'CNN':
        model = SimpleCNN()
    else:
        raise ValueError("Currently only CNN is implemented")
    
    return model.to(Config.DEVICE)
