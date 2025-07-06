import torch
from src.model import Net
from src.dataset import get_data_loaders
from src.train import train_model

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    model = Net()
    train_model(model, train_loader, test_loader)
