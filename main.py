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
            v.append(r[:-1])
            y.append(r[-1])

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
    optimizer = optim.SGD(net.parameters(), lr=1.5)
    # print(torch.tensor(x).shape)
    # print(N)
    x_train = torch.tensor(x)[:N-3000].cuda()
    y_train = torch.tensor(y_train.transpose(), dtype=torch.long)[:N-3000].cuda()

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i in range(4000):
            optimizer.zero_grad()

            outputs = net(x_train)
            loss = criterion(outputs.cuda(), y_train.cuda())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 5 == 4:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss))
                running_loss = 0.0

            optimizer.step()

    # Testing
    x_test = x[N-3000: N].cuda()
    y_test= y[N-3000: N].cuda()
    
    running_test_loss = 0.0
    for i in range(N-3000, N):
        test_outputs= net(x_test)
        cross_val=criterion(test_outputs.cuda(), y_test)
        running_test_loss += cross_val.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print(
                (i + 1, running_test_loss))

        

if __name__ == "__main__":
    main()