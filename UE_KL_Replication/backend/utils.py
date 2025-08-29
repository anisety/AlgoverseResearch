import torch
from torch.utils.data import DataLoader, TensorDataset

def get_dataloader(batch_size=16):
    # Example synthetic dataset
    X = torch.randn(100, 10)  # 100 samples, 10 features
    Y_target = torch.randn(100, 5)  # target distributions over 5 classes
    dataset = TensorDataset(X, Y_target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
