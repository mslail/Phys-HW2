import csv
import numpy as np
import json, argparse, torch
import torch.optim as optim
import torch.nn as nn
from nn_gen import Net

def main():
    # Reading Arguments
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--param', metavar='param.json',
                        help='parameter file name')
    args = parser.parse_args()

    # Opening JSON config file
    with open(args.param) as paramfile:
        param = json.load(paramfile)

    # Reading Data from CSV
    v = []
    y = []
    with open('even_mnist.csv', 'r') as f:
        reader = csv.reader(f)
        for data in reader:
            r = data[0].split()
            v.append(r[:-1])
            y.append(r[-1])
    
    # Reshaping Data
    x = []
    for row in v:
        imageReshape = np.reshape(row, (1,14,14))
        x.append(imageReshape)

    # Initializing Np arrays
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Model
    net = Net()

    # Params
    epochs = param['epochs']
    learning_rate = param['learning_rate']
    N = len(x)

    # Loss
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # Training Data
    x_train = torch.tensor(x)[:N-3000]
    y_train = torch.tensor(y.transpose(), dtype=torch.long)[:N-3000]

    # Testing Data
    x_test = torch.tensor(x)[N-3000:]
    y_test= torch.tensor(y.transpose(), dtype=torch.long)[N-3000:]

    print('Attempting to start training')

    # Training
    for epoch in range(epochs):
        # Back Prop
        train_val= net.backprop(x_train, y_train, criterion, epoch, optimizer)
        # Testing 
        test_val= net.test(x_test, y_test, criterion, epoch)

        if not ((epoch + 1) % 5):
                print('Epoch [{}/{}]'.format(epoch+1, epochs)+\
                        '\tTraining Loss: {:.4f}'.format(train_val)+\
                        '\tTest Loss: {:.4f}'.format(test_val))

    # Testing classification
    outputs = net(x_test)
    i, predicted = torch.max(outputs, 1)
    print('Index: ', i)
    print('outputs: ', predicted)
    print('Training Value: ', train_val)
    print('Test Value: ', test_val)

if __name__ == "__main__":
    main()
