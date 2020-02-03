import torch
import torch.nn as nn
import torch.nn.functional as func
from scipy import signal, ndimage
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(7, 14, 3)
        self.fc1= nn.Linear(14*2*2, 120)
        self.fc2= nn.Linear(120, 32)
        self.fc3= nn.Linear(32, 10)
    
    def forward(self, x):
        # print(self.conv1(x)[0])
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))

        x = x.view(x.size(0), 14*2*2)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def backprop(self, x_train, y_train, loss, epoch, optimizer):
        self.train()
        inputs= torch.from_numpy(x_train)
        targets= torch.from_numpy(y_train)
        print(inputs)
        outputs= self(inputs)
        obj_val= loss(self.forward(inputs).reshape(-1), targets)
        optimizer.zero_grad() 
        obj_val.backward()
        return obj_val.item()

    def test(self, data, loss, epoch):
        self.eval()
        with torch.no_grad():
            inputs= torch.from_numpy(data.x_test)
            targets= torch.from_numpy(data.y_test)
            outputs= self(inputs)
            cross_val= loss(self.forward(inputs).reshape(-1), targets)
        return cross_val.item()
        