import time
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, TensorDataset

import nn_torch_mnist
import torch_mnist

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out'

def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]

#np.random.seed(100) 
#np.random.seed(int(time.time() * 1e6) % 2**31)

# Load
ds_train, ds_test = MNIST(DATA_DIR, train=True, download=True), MNIST(DATA_DIR, train=False)
train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float32) / 255
train_y = ds_train.targets.numpy()
train_x, valid_x = train_x[:55000], train_x[55000:]
train_y, valid_y = train_y[:55000], train_y[55000:]
test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float32) / 255
test_y = ds_test.targets.numpy()

# Normalize
train_mean = train_x.mean()
train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))

# One-Hot
train_y, valid_y, test_y = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))

# Convert to tensors
torch_train_x = torch.tensor(train_x, dtype=torch.float32)
torch_train_y = torch.tensor(train_y, dtype=torch.float32)
torch_valid_x = torch.tensor(valid_x, dtype=torch.float32)
torch_valid_y = torch.tensor(valid_y, dtype=torch.float32)
torch_test_x = torch.tensor(test_x, dtype=torch.float32)
torch_test_y = torch.tensor(test_y, dtype=torch.float32)

# DataLoader objects
batch_size = 50
train_loader = DataLoader(TensorDataset(torch_train_x, torch_train_y), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(TensorDataset(torch_valid_x, torch_valid_y), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(torch_test_x, torch_test_y), batch_size=batch_size, shuffle=False)

def init_model_arch():
    return torch_mnist.ConvolutionalModel()

config = {
    'max_epochs': 8,
    'batch_size': batch_size,
    'save_dir': SAVE_DIR,
    'weight_decay': 1e-2,
    'lr_policy': {1: {'lr': 1e-1}, 3: {'lr': 1e-2}, 5: {'lr': 1e-3}, 7: {'lr': 1e-4}}
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for wd in [1e-2]:
    config["weight_decay"] = wd
    model = init_model_arch().to(device)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    print(f"Testing model regularization factor: {wd}")
    nn_torch_mnist.train(model, train_loader, valid_loader, config, device, use_optim_reg=True)
    nn_torch_mnist.evaluate("Test", model, test_loader, device)

