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

data = dataLoader.main()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# labels = []
# inputPaths = []


# for i, d in enumerate(data, 0):
#     inputPaths = d
#     print(inputPaths)

# take one batch
inputBatch = data[2]

# split the batch into labels and input paths
unzipped = [[i for i, j in inputBatch],
            [j for i, j in inputBatch]]
# print(unzipped)

labels, inputPaths = unzipped
# print(f'inputs: {inputPaths} labels: {labels}')

# creates new inputs list
inputs = []

# for every path in inputPaths, load the image.
# If it is a 'no' image then it has a opens with pillow
# Otherwise it uses the same returnImage function from matlabReader
for i in inputPaths:
    if '.mat' in str(i):
        inputs = inputs + [returnImage(i)]
    elif '.png' in str(i):
        inputs = inputs + [Image.open(str(i).lstrip("b'").rstrip("'"))]
print(inputs)

trans = transforms.ToTensor()
nums = [0, 1, 2, 3]
ninputs = torch.Tensor([])

# for i, d in enumerate(inputs, 0):
#     ninputs = torch.cat((ninputs, trans(d)), i)

print(f'1: {ninputs, inputs[0]}')
i1 = torch.cat((ninputs, trans(inputs[0])), 0)
print(f'2: {i1, inputs[1]}')
i2 = torch.cat((i1, trans(inputs[1])), 0)
print(f'3: {i2}')

labels = torch.FloatTensor(labels)

# print(f'inputs: {inputs[1]}')
print(f'ninputs: {i2, i2.shape}')
print(f'label: {labels}')
