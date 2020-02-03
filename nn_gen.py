import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(7, 14, 3)
        self.fc1= nn.Linear(14*2*2, 84)
        self.fc2= nn.Linear(84, 14)
        self.fc3= nn.Linear(14, 9)
    
    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
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
        outputs= self(x_train)
        obj_val= loss(self.forward(x_train), y_train)
        optimizer.zero_grad() 
        obj_val.backward()
        optimizer.step()
        return obj_val.item()

    def test(self, x_test, y_test, loss, epoch):
        self.eval()
        with torch.no_grad():
            # inputs= torch.from_numpy(x_test)
            # targets= torch.from_numpy(y_test)
            outputs= self(x_test)
            cross_val= loss(outputs, y_test)
        return cross_val.item()
        
