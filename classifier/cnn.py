from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from dataLoaderFile.matlabReader import returnImage
import numpy as np

from dataLoaderFile import dataLoader
# import dataLoader

import torch.optim as optim

from tqdm import tqdm

import os

from dataLoaderFile.customDataset import dataset


# data = dataLoader.main()


# defining network class and passing in nn.Module, a package that includes all the neural network functionality


class Net(nn.Module):
    # constructor
    def __init__(self):
        # immediately call the super class
        super(Net, self).__init__()
        # define network layers
        # 2d convolutional layers (input channels, output channels, kernel size)
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.conv2 = nn.Conv2d(3, 5, 5)
        self.conv3 = nn.Conv2d(5, 5, 2)
        # Pooling layer (kernel size, step)
        self.pool = nn.MaxPool2d(2, 2)
        # linear layers (input features, output features)
        self.fc1 = nn.Linear(5 * 62 * 62, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 4)
        # ends with 4 features, one for each type of cancer and one for 'no'

    # forward propagation function
    def forward(self, x):
        # pass through layer, rectified linear function and pool all at once
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # print(x[0].shape)
        # transform into linear form
        x = x.view(-1, 5 * 62 * 62)
        # print(x[0].shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # return x
        return F.softmax(x, dim=1)


# instantiate network
net = Net()

# define loss function and optimiser, will be useful later
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

# zero the gradient
optimizer.zero_grad()
net.zero_grad()

# transformt the network into datatype double so that it is consistent with the data
# net = net.float()
net = net.double()
# net = net.int()


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
    # batchImages[0].show()
    # adding on the rest of the images transformed as tensors
    # torch.cat concatenates tensors in a specified dimension
    for i in range(1, len(batchImages)):
        img = batchImages[i]
        # print(img.size)
        # resize images if they are not 512x512
        if img.size != (512, 512):
            # print('resized')
            img = img.resize((512, 512))
            # print(img.size)
        # print(transImages.shape, trans(img).shape)
        transImages = torch.cat((transImages, trans(img)), 0)

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
        print(m)
        # add the batch to the existing list of batches of batches
        x = x + [n]
        # output batches need to be made into a datatype 'long tensor' as this is what the loss function expects
        # also every value in the output needs to be shifted by 1 in order to account for the 0th index
        # pytorch treats the first index as 0 but we started from 1 in our data for ease of understanding
        y = y + [m.long() - 1]
    return x, y


def train(X, Y):
    # Epochs are the number of large loops through the data you do
    EPOCHS = 100
    for epoch in range(EPOCHS):
        # for every colllection of batches
        # for i in tqdm(range(len(X))):
        # print(epoch)
        for i in range(len(X)):
            # take the current batch
            batch_X = X[i]
            batch_Y = Y[i]

            # zero the gradients
            # net.zero_grad()
            # push through the network
            outputs = net(batch_X)

            # print(outputs.shape)
            # print(outputs)

            # Calculate loss
            # loss = loss accumulator
            loss = criterion(outputs, batch_Y)

            # print(outputs, batch_Y, lossAcc)
            # print(lossAcc)

            # back propagate
            optimizer.zero_grad()
            loss.backward()
            # step down the loss function

            optimizer.step()
        if epoch % 100 == 0:
            print(epoch, loss)
    # print('99', lossAcc)


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
                # print(net_out[x])
                # take the highest probability in the output
                predicted_class = torch.argmax(net_out[x])
                print(f'predicted: {predicted_class}, real: {test_Y[i][x]}')
                # check if output class matches the real class
                if predicted_class == real_class:
                    # increment correct if it matches
                    correct += 1
                # increment total
                total += 1
    print(correct, total)
    # output accuracy percentage to 3sf
    print('Accuracy: ', round(correct/total, 3))


def loadSkimages():
    filepath = 'classifier/dataLoaderFile/NEA_data/extracted/skimages'
    x = []
    y = []
    for filename in os.listdir(filepath):
        if not('.txt' in filename):
            fullpath = filepath + '/' + str(filename) + '/'
            for attribute in os.listdir(fullpath):
                # print(attribute)
                if attribute == filename + '.jpg':
                    img = Image.open(fullpath + '/' + attribute)
                    x = x + [trans(img)]
                if attribute == filename + 'result.txt':
                    file = open(fullpath + '/' + attribute)
                    y = y + [float(file.read())]
                    file.close()
    # img = Image.open('classifier/dataLoaderFile/NEA_data/extracted/skimages/2no.jpg')
    # img.show()
    # print(img)
    print(len(x))
    # print(y)
    return(x, y)

# def returnskiBatch(start):


def loadskiBatches(skidata, numOfBatches, batchSize, start=0, labels='normal'):
    # i = start
    batches = [skidata[batchSize*i:batchSize*(i+1)]
               for i in range(int(numOfBatches))]
    if labels == 'labels':
        # print(labels)
        for x in range(len(batches)):
            batches[x] = torch.Tensor(batches[x]).long() - 1
    if labels == "images":
        for x in range(len(batches)):
            concatenated = (batches[x][0].unsqueeze(0))
            # print(concatenated.shape)
            for y in range(1, len(batches[x])):
                # print(y)
                # print(batches[x][y].unsqueeze(0).shape)
                concatenated = torch.cat(
                    (concatenated, batches[x][y].unsqueeze(0)), 0)
            batches[x] = concatenated

    return(batches)


# hyperparameters
numOfBatches = 1
batchSize = 4
epochs = 4

dataSet = dataset(csv_file='classifier/dataLoaderFile/NEA_data/extracted/randomPaths.csv',
                  root_dir='', transforms=transforms.ToTensor())
print(len(dataSet))
train_set, test_set = torch.utils.data.random_split(dataSet, [800, 199])
train_loader = DataLoader(
    dataset=train_set, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batchSize, shuffle=True)


def main():

    # X, Y = loadBatches(numOfBatches)
    # print(len(X), Y)
    # for x in X:
    #     print(x.shape)
    # train(X, Y)
    # print(Y, len(Y))

    # test the network on the last 5 batches in the data
    # test_X, test_Y = loadBatches(5, start = 30)
    # print(len(data))
    # test(test_X, test_Y)
    # for x in data:
    #     for y in x:
    #         print(y)

    # X, Y = loadBatches(1)
    # train(X, Y)

    # skimages
    x, y = loadSkimages()

    X = loadskiBatches(x, numOfBatches, batchSize, labels='images')
    Y = loadskiBatches(y, numOfBatches, batchSize, labels='labels')
    print(y)
    print(len(X), Y)
    # for x in X:
    #     print(x.shape)
    # train(X, Y)

    for epoch in range(epochs):
        losses = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            # print(data)
            output = net(data)
            loss = criterion(output, targets)

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step
        print(f'Cost as {epoch} = {sum(losses)/len(losses)}')

    def accuracy(loader, net):
        num_correct = 0
        num_samples = 0
        net.eval()

        with torch.no_grad():
            for x, y in loader:
                out = net(x)
                num_correct += (out == y).sum()
                num_samples += out.size(0)

            print(
                f'{num_correct} right out of {num_samples}.  {num_correct/num_samples}')

        net.train()

    print('Training data')
    accuracy(train_loader, net)
    print('Test data')
    accuracy(test_loader, net)


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
