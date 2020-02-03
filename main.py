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

    y = np.array(y, dtype=np.float32)
    #print(y_train)
    net = Net()
    x = []
    for row in v:
        imageReshape = np.reshape(row, (1,14,14))
        x.append(imageReshape)
    N = len(x)
    #print(x[0])
    x = np.array(x, dtype=np.float32)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # print(net.parameters())
    optimizer = optim.SGD(net.parameters(), lr=0.15)
    # print(torch.tensor(x).shape)
    # print(N)
    x_train = torch.tensor(x)[:N-3000]
    y_train = torch.tensor(y.transpose(), dtype=torch.long)[:N-3000]

        # Testing
    x_test = torch.tensor(x[N-3000: N])
    y_test= torch.tensor(y.transpose(), dtype=torch.long)[N-3000: N]
    net.train()
    # for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for n in range(2):
        print("Training Data itteration: {}".format(n))
        for epoch in range(75):
            train_val= net.backprop(x_train, y_train, criterion, epoch, optimizer)
            test_val= net.test(x_test, y_test, criterion, epoch)
            if not ((epoch + 1) % 5):
                    print('Epoch [{}/{}]'.format(epoch+1, 75)+\
                            '\tTraining Loss: {:.4f}'.format(train_val)+\
                            '\tTest Loss: {:.4f}'.format(test_val))

    print('Training Value: ', train_val)
    print('Test Value: ', test_val)
        # optimizer.zero_grad()

        # outputs = net(x_train)
        # loss = criterion(outputs, y_train)
        # loss.backward()
        # optimizer.step()

        # running_loss += loss.item()
        # if i % 5 == 4:    # print every 5 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #         (epoch + 1, i + 1, running_loss / 5))
        #     running_loss = 0.0

        # optimizer.step()


    # net.eval()

    # for epoch in range(2):
    #     running_test_loss = 0.0
    #     for i in range(3000):
    #         with torch.no_grad():
    #             test_outputs= net(x_test)
    #             cross_val=criterion(test_outputs, y_test)
    #             running_test_loss += cross_val.item()
    #             print(cross_val.item())
    #             if i % 5 == 4:    # print every 5 mini-batches
    #                 print('[%d, %5d] loss: %.3f' %
    #                 (epoch + 1, i + 1, running_test_loss / 5))
    #                 running_test_loss = 0.0        

if __name__ == "__main__":
    main()
