import csv
import numpy as np
import json, argparse, torch
import torch.optim as optim
import torch.nn as nn
from nn_gen import Net



def main():
    v = []
    y = []
    with open('even_mnist.csv', 'r') as f:
        reader = csv.reader(f)
        for data in reader:
            r = data[0].split()
            x = r[:-1]
            v.append(x)
            y.append(r[-1])
        
    #x_train = np.array(v, dtype=np.float32)
    y_train = np.array(y, dtype=np.float32)
    #print(y_train)
    net = Net().cuda()
    x = []
    for row in v:
        imageReshape = np.reshape(row, (1,14,14))
        x.append(imageReshape)
    N = len(x)
    #print(x[0])
    x = np.array(x, dtype=np.float32)
    criterion = nn.CrossEntropyLoss()
    # print(net.parameters())
    optimizer = optim.SGD(net.parameters(), lr=0.5)
    # print(torch.tensor(x).shape)
    # print(N)
    x = torch.tensor(x)
    y = torch.tensor(y_train.transpose(), dtype=torch.long)

    print(x.device)

    
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i in range(N - 3000):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = (x, y)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.cuda())
            loss = criterion(outputs.cuda(), labels.cuda())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 5 == 4:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0
            # optimizer.step()

if __name__ == "__main__":
    main()