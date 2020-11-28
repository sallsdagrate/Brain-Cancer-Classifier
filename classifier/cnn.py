from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from dataLoaderFile.matlabReader import returnImage
import numpy as np

from dataLoaderFile import dataLoader

import torch.optim as optim

from tqdm import tqdm


data = dataLoader.main()

# defining network class and passing in nn.Module, a package that includes all the neural network functionality


class Net(nn.Module):
    # constructor
    def __init__(self):
        # immediately call the super class
        super(Net, self).__init__()
        # define network layers
        # 2d convolutional layers (input channels, output channels, kernel size)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Pooling layer (kernel size, step)
        self.pool = nn.MaxPool2d(2, 2)
        # linear layers (input features, output features)
        self.fc1 = nn.Linear(16 * 125 * 125, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 4)
        # ends with 4 features, one for each type of cancer and one for 'no'

    # forward propagation function
    def forward(self, x):
        # pass through layer, rectified linear function and pool all at once
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x[0].shape)
        # transform into linear form
        x = x.view(-1, 16 * 125 * 125)
        # print(x[0].shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


# instantiate network
net = Net()

# define loss function and optimiser, will be useful later
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# zero the gradient
optimizer.zero_grad()
net.zero_grad()

# transformt the network into datatype double so that it is consistent with the data
net = net.double()


# premade function to transform image into pytorch tensor
trans = transforms.ToTensor()


def translateBatch(newBatch):

    # newBatch is in the form [label, image] so we need to split them up
    # this splits the batch into labels and input paths but zipped
    unzipped = [[i for i, j in newBatch],
                [j for i, j in newBatch]]

    # splitting unzipped into labels and images
    labels, inputPaths = unzipped

    # creates new inputs list
    batchImages = []

    # for every path in inputPaths, load the image.
    # If it is a 'no' image then it has a opens with pillow
    # Otherwise it uses the same returnImage function from matlabReader
    for i in inputPaths:
        if '.mat' in str(i):
            batchImages = batchImages + [returnImage(i)]
        elif '.png' in str(i):
            # path needs to be adjusted slightly to work and the image must be translated into image mode 'I' like the rest of them
            batchImages = batchImages + [Image.open(str(i).lstrip("b'").rstrip("'")).convert(mode='I')]

    # initialising ninputs as the first image transformed into a tensor
    # so that everything else can be added onto it
    transImages = trans(batchImages[0])
    
    # adding on the rest of the images transformed as tensors
    # torch.cat concatenates tensors in a specified dimension
    for i in range(1, len(batchImages)):
        print(transImages.shape, trans(batchImages[0]).shape)
        transImages = torch.cat((transImages, trans(batchImages[i])), 0)

    # labels is just a list so can be made directly into a tensor
    labels = torch.FloatTensor(labels)

    # print(f'ninputs shape: {transImages.shape}')
    # print(f'label: {labels}')

    return transImages.unsqueeze(1).double(), labels


# function to return a specified number of batches with the option of starting at a certain index

def loadBatches(numOfBatches, start=0):
    # initialise a list of x (inputs) and y (expected output)
    x = []
    y = []
    # iterated through the data for a specified number of batches from either the 0th index or from a specified index
    for i in range(start, start + numOfBatches):
        # return a batch
        n, m = translateBatch(data[i])
        # add the batch to the existing list of batches of batches
        x = x + [n]
        # output batches need to be made into a datatype 'long tensor' as this is what the loss function expects
        # also every value in the output needs to be shifted by 1 in order to account for the 0th index
        # pytorch treats the first index as 0 but we started from 1 in our data for ease of understanding
        y = y + [m.long() - 1]
    return x, y



def train(X, Y):
    # Epochs are the number of large loops through the data you do
    EPOCHS = 1
    for epoch in range(EPOCHS):
        # for every colllection of batches
        for i in tqdm(range(len(X))):
            # take the current batch
            batch_X = X[i]
            batch_Y = Y[i]

            # zero the gradients
            net.zero_grad()
            # push through the network
            outputs = net(batch_X)

            # print(outputs.shape)
            # print(outputs)

            # Calculate loss
            # lossAcc = loss accumulator
            lossAcc = loss(outputs, batch_Y)

            print(outputs, batch_Y, lossAcc)

            # back propagate
            lossAcc.backward()
            # step down the loss function
            optimizer.step()


def test(test_X, test_Y):
    # initialise the statistics as 0
    correct = 0
    total = 0
    # no gradients need to be calculated for the verification process
    with torch.no_grad():
        # for each batch in the test data
        for i in tqdm(range(len(test_X))):
            # push test data through the network
            net_out = net(test_X[i])
            # for every image
            for x in range(len(test_Y[i])):

                real_class = test_Y[i][x]
                print(net_out[x])
                # take the highest probability in the output
                predicted_class = torch.argmax(net_out[x])
                print(predicted_class, test_Y[i][x])
                # check if output class matches the real class
                if predicted_class == real_class:
                    # increment correct if it matches
                    correct += 1
                # increment total
                total += 1
    print(correct, total)
    # output accuracy percentage to 3sf
    print('Accuracy: ', round(correct/total, 3))


def main():

    numOfBatches = 30
    # X, Y = loadBatches(numOfBatches)
    # train(X, Y)
    # print(Y, len(Y))
    
    # test the network on the last 5 batches in the data
    test_X, test_Y = loadBatches(5, start=len(data)-6)
    # test(test_X, test_Y)
    # for x in data:
    #     for y in x:
    #         print(y)

main()
# output = net(n)
# print(output.shape)
# print(net)

# .view(-1, 512*512)


# # take one batch
# inputBatch = data[2]

# # split the batch into labels and input paths
# unzipped = [[i for i, j in inputBatch],
#             [j for i, j in inputBatch]]
# # print(unzipped)

# # splitting unzipped into labels and images
# labels, inputPaths = unzipped
# # print(f'inputs: {inputPaths} labels: {labels}')

# # creates new inputs list
# inputs = []

# # for every path in inputPaths, load the image.
# # If it is a 'no' image then it has a opens with pillow
# # Otherwise it uses the same returnImage function from matlabReader
# for i in inputPaths:
#     if '.mat' in str(i):
#         inputs = inputs + [returnImage(i)]
#     elif '.png' in str(i):
#         inputs = inputs + [Image.open(str(i).lstrip("b'").rstrip("'"))]
# print(inputs)


# # initialising ninputs as the first image transformed into a tensor so that everything else can be added onto it
# ninputs = trans(inputs[0])

# # print(f'1: {ninputs.shape, inputs[1]}')
# # i1 = torch.cat((ninputs, trans(inputs[0])), 0)
# # print(f'2: {i1.shape, inputs[2]}')
# # i2 = torch.cat((i1, trans(inputs[2])), 0)
# # print(f'1: {i2.shape, inputs[3]}')
# # i3 = torch.cat((i2, trans(inputs[3])), 0)
# # print(f'2: {i3.shape}')

# # print(f'0 {ninputs}')

# # adding on the rest of the images transformed as tensors
# # torch.cat concatenates tensors in a specified dimension
# for i in range(1, len(inputs)):
#     ninputs = torch.cat((ninputs, trans(inputs[i])), 0)
#     print(f'{i} ', inputs[i])

# print(f'ninputs shape: {ninputs.shape}')

# # labels is just a list so can be made directly into a tensor
# labels = torch.FloatTensor(labels)

# # print(ninputs, i3,torch.sub(ninputs, i3), trans(inputs))
# # print(f'inputs: {inputs[1]}')
# # print(f'ninputs: {i2, i2.shape}')
# print(f'label: {labels}')


# # outputs = net(ninputs.view(-1, 512*512))
# # print(outputs)
