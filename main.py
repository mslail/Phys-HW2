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
    x = []
    for row in v:
        imageReshape = np.reshape(row, (1,14,14))
        x.append(imageReshape)
    N = len(x)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    net = Net()

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr=0.15)
    x_train = torch.tensor(x)[:N-3000]
    y_train = torch.tensor(y.transpose(), dtype=torch.long)[:N-3000]

    # Testing
    x_test = torch.tensor(x)[N-3000: N]
    y_test= torch.tensor(y.transpose(), dtype=torch.long)[N-3000: N]
    net.train()

    for n in range(2):
        print("Training Data itteration: {}".format(n))
        for epoch in range(75):
            train_val= net.backprop(x_train, y_train, criterion, epoch, optimizer)
            test_val= net.test(x_test, y_test, criterion, epoch)
            if not ((epoch + 1) % 5):
                    print('Epoch [{}/{}]'.format(epoch+1, 75)+\
                            '\tTraining Loss: {:.4f}'.format(train_val)+\
                            '\tTest Loss: {:.4f}'.format(test_val))
    outputs = net(x_test)
    i, predicted = torch.max(outputs, 1)
    print('Index: ', i)
    print('outputs: ', predicted)
    print('Training Value: ', train_val)
    print('Test Value: ', test_val)

if __name__ == "__main__":
    main()
