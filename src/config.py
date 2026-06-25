import torch

class Config:
    # Environment
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    DATASET = 'CIFAR10' # Choose 'MNIST' or 'CIFAR10'
    DATA_DIR = './data'
    
    # Training
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # STAM Landscape Parameters
    SPARSE_GRID_RESOLUTION = 7 
    RENDER_RESOLUTION = 100 
    GRID_MARGIN = 0.5 
    
    # Visualization Settings
    LIVE_TRAINING = True
    COLORMAP = 'coolwarm'
    
    # Model
    MODEL_TYPE = 'CNN'
    
    # Output
    PLOT_DIR = './visualization'
