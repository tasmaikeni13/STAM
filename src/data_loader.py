import os
import tarfile
import requests
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .config import Config

def download_cifar10_from_mirror(data_dir):
    url = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    cifar_dir = os.path.join(data_dir, "cifar-10-batches-py")
    
    if not os.path.exists(cifar_dir):
        if not os.path.exists(tar_path):
            print(f"Downloading CIFAR-10 from {url}...")
            response = requests.get(url, stream=True)
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        
        print("Extracting CIFAR-10...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Extraction complete.")

def get_dataloaders():
    if Config.DATASET == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root=Config.DATA_DIR, train=True, download=True, transform=transform)
        val_dataset = torchvision.datasets.MNIST(root=Config.DATA_DIR, train=False, download=True, transform=transform)
        
    elif Config.DATASET == 'CIFAR10':
        download_cifar10_from_mirror(Config.DATA_DIR)
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(root=Config.DATA_DIR, train=True, download=False, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10(root=Config.DATA_DIR, train=False, download=False, transform=transform_test)
    
    else:
        raise ValueError("Unknown dataset")

    # num_workers=0 to prevent issues on windows multiprocessing
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, val_loader
