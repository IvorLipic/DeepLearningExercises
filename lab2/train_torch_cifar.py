import os
import pickle
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

import nn_torch_cifar
import torch_cifar

def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict

DATA_DIR = Path(__file__).parent / 'datasets' / 'CIFAR'

img_height = 32
img_width = 32
num_channels = 3
num_classes = 10

train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
  subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
  train_x = np.vstack((train_x, subset['data']))
  train_y += subset['labels']
train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
data_mean = train_x.mean((0, 1, 2))
data_std = train_x.std((0, 1, 2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

# Convert from NHWC to NCHW for PyTorch
train_x = train_x.transpose(0, 3, 1, 2)
valid_x = valid_x.transpose(0, 3, 1, 2)
test_x = test_x.transpose(0, 3, 1, 2)

def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]

train_y = dense_to_one_hot(train_y, num_classes)
valid_y = dense_to_one_hot(valid_y, num_classes)
test_y = dense_to_one_hot(test_y, num_classes)

torch_train_x = torch.tensor(train_x, dtype=torch.float32)
torch_train_y = torch.tensor(train_y, dtype=torch.float32)
torch_valid_x = torch.tensor(valid_x, dtype=torch.float32)
torch_valid_y = torch.tensor(valid_y, dtype=torch.float32)
torch_test_x = torch.tensor(test_x, dtype=torch.float32)
torch_test_y = torch.tensor(test_y, dtype=torch.float32)

batch_size = 50
train_loader = DataLoader(TensorDataset(torch_train_x, torch_train_y), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(TensorDataset(torch_valid_x, torch_valid_y), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(torch_test_x, torch_test_y), batch_size=batch_size, shuffle=False)

SAVE_DIR = Path(__file__).parent / 'out_cifar'

def init_model_arch():
    return torch_cifar.CIFARModel()

config = {
    'max_epochs': 50,
    'batch_size': batch_size,
    'save_dir': str(SAVE_DIR),
    'weight_decay': 1e-3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = init_model_arch().to(device)
print("Total parameters:", sum(p.numel() for p in model.parameters()))
print(f"Regularization factor: {config['weight_decay']}")
nn_torch_cifar.train(model, train_loader, valid_loader, config, device, data_mean, data_std, use_hinge_loss=False)