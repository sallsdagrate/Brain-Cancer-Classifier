from __future__ import print_function
from skimage import data, io, filters
# from cnn import Net
from dataLoaderFile import dataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import torch
from PIL import Image, ImageEnhance
from numpy import asarray
from dataLoaderFile import matlabReader
import os

from numpy import genfromtxt
# img = matlabReader.returnImage('classifier/dataLoaderFile/NEA_data/extracted/300.mat')


import matplotlib.pyplot as plt

from tqdm import tqdm


# defining network class and passing in nn.Module, a package that includes all the neural network functionality


class Net(nn.Module):
    # constructor
    def __init__(self):
        # immediately call the super class
        super(Net, self).__init__()
        # define network layers
        # 2d convolutional layers (input channels, output channels, kernel size)
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.conv2 = nn.Conv2d(5, 5, 5)
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

# load parameters
net.load_state_dict(torch.load('model - 100.pth'))

trans = transforms.ToTensor()


def push(path):
    path = path.lstrip("b'").rstrip("'")
    if '.mat' in path:
        img = matlabReader.returnImage(path)
    else:
        img = Image.open(path)
        img = img.convert(mode='I')

    if img.size != (512, 512):
        img = img.resize((512, 512))

    # save it in colour scheme
    image = asarray(img)
    plt.imsave('imagetest.jpg', image)

    # reopen and convert
    img = Image.open('imagetest.jpg')
    img = asarray(img)
    img = trans(img)
    img = img.unsqueeze(0)
    # print(img)
    # push
    output = net.forward(img)

    # find the class with the highest probability
    output, scan_class = torch.max(output, 1)

    scan_class = scan_class.item()

    return scan_class


testData = genfromtxt(
    'classifier/dataLoaderFile/NEA_data/extracted/randomPaths copy.csv', delimiter=',', dtype=None)
print(len(testData))
testData = testData[2827:]
print(len(testData))
# print(testData)
testData = [[i for i, j in testData],
            [j for i, j in testData]]

correct = 0
total = 0
# count = 100
for i in tqdm(range(len(testData[0]))):
    result = push(str(testData[1][i]))
    if result + 1 == testData[0][i]:
        correct += 1
    total += 1

print(correct, total)
print('Accuracy: ', round(correct/total, 5))
