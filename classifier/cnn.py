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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 125 * 125, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x[0].shape)
        x = x.view(-1, 16 * 125 * 125)
        # print(x[0].shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


net = Net()

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

optimizer.zero_grad()

net.zero_grad()

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
            batchImages = batchImages + \
                [Image.open(str(i).lstrip("b'").rstrip("'")).convert(mode='I')]

    # initialising ninputs as the first image transformed into a tensor
    # so that everything else can be added onto it
    transImages = trans(batchImages[0])

    # adding on the rest of the images transformed as tensors
    # torch.cat concatenates tensors in a specified dimension
    for i in range(1, len(batchImages)):
        transImages = torch.cat((transImages, trans(batchImages[i])), 0)

    # labels is just a list so can be made directly into a tensor
    labels = torch.FloatTensor(labels)

    # print(f'ninputs shape: {transImages.shape}')
    # print(f'label: {labels}')

    return transImages.unsqueeze(1).double(), labels

# function to return a specified number of batches with the option of starting at a certain index


def loadBatches(numOfBatches, start=0):
    x = []
    y = []
    for i in range(start, start + numOfBatches):
        n, m = translateBatch(data[i])
        x = x + [n]
        y = y + [m.long() - 1]
    return x, y


numOfBatches = 15
X, Y = loadBatches(numOfBatches)


EPOCHS = 2
for epoch in range(EPOCHS):
    for i in tqdm(range(len(X))):
        batch_X = X[i]
        batch_Y = Y[i]

        net.zero_grad()
        outputs = net(batch_X)

        # lossAcc = loss accumulator
        lossAcc = loss(outputs, batch_Y)
        print(outputs, batch_Y, lossAcc)

        lossAcc.backward()
        optimizer.step()
# print(loss)

correct = 0
total = 0
test_X, test_Y = loadBatches(5, len(data) - 5)
# print(test_Y[0])
# print(test_X[0])
# out = net(test_X[0])
# print(out)
# print(len(test_Y))
# print(len(test_X))
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        net_out = net(test_X[i])
        for x in range(len(test_Y[i])):
            real_class = test_Y[i][x]
            print(test_Y[i][x], net_out[x])
            predicted_class = torch.argmax(net_out[x])
            print(predicted_class)
            if predicted_class == real_class:
                correct += 1
            total += 1
print(correct, total)
print('Accuracy: ', round(correct/total, 3))


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
