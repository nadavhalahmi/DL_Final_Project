import os
import numpy as np
import sklearn
import torch
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.transforms
import torch.optim

root_dir = os.path.expanduser('~/.pytorch-datasets/')


tf_ds = torchvision.transforms.ToTensor()

batch_size = 512
train_size = batch_size * 10
test_size = batch_size * 2

# Datasets and loaders
ds_train = torchvision.datasets.MNIST(root=root_dir, download=True, train=True, transform=tf_ds)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size,
                                       sampler=torch.utils.data.SubsetRandomSampler(range(0,train_size)))
ds_test =  torchvision.datasets.MNIST(root=root_dir, download=True, train=False, transform=tf_ds)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size,
                                       sampler=torch.utils.data.SubsetRandomSampler(range(0,test_size)))

x0, y0 = ds_train[0]
n_features = torch.numel(x0)
n_classes = 10

print(f'x0: {x0.shape}, y0: {y0}')


class MLP(torch.nn.Module):
    NLS = {'relu': torch.nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax,
           'logsoftmax': nn.LogSoftmax}

    def __init__(self, D_in: int, hidden_dims: list, D_out: int, nonlin='relu'):
        super().__init__()

        all_dims = [D_in, *hidden_dims, D_out]
        layers = []

        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            layers += [
                nn.Linear(in_dim, out_dim, bias=True),
                MLP.NLS[nonlin]()
            ]

        self.fc_layers = nn.Sequential(*layers[:-1])
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        z = self.fc_layers(x)
        y_pred = self.log_softmax(z)
        return y_pred


# Loss:
# Note: NLLLoss assumes *log*-probabilities (given by our LogSoftmax layer)
loss_fn = nn.NLLLoss()

# Model for training
model = MLP(D_in=n_features, hidden_dims=[32, 32, 32], D_out=n_classes, nonlin='relu')

# Optimizer over our model's parameters
optimizer = torch.optim.SGD(params=model.parameters(), lr=5e-2, weight_decay=0.01, momentum=0.9)

num_epochs = 10
for epoch_idx in range(num_epochs):
    total_loss = 0

    for batch_idx, (X, y) in enumerate(dl_train):
        # Forward pass
        y_pred = model(X)

        # Compute loss
        loss = loss_fn(y_pred, y)
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()  # Zero gradients of all parameters
        loss.backward()  # Run backprop algorithms to calculate gradients

        # Optimization step
        optimizer.step()  # Use gradients to update model parameters

    print(f'Epoch #{epoch_idx + 1}: Avg. loss={total_loss / len(dl_train)}')