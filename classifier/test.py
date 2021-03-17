# import torch
# import torchvision
# import torchvision.transforms as transforms

# import torch.nn as nn
# import torch.nn.functional as F

# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# output = loss(input, target)
# output.backward()
# print(input, target, output)


from dataLoaderFile import matlabReader
from PIL import Image

matlabReader.extractImage(
    '18.mat', 'classifier/dataLoaderFile/NEA_data/extracted/18.mat')
