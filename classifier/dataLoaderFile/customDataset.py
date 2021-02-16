import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from dataLoaderFile.matlabReader import returnImage
from skimage import data, io, filters
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import asarray
# ^^ necessary imports

# define dataset class using prebuilt library


class dataset (Dataset):

    def __init__(self, csv_file, root_dir, transforms=None):
        # initialising the place to read the labels/classes
        self.annotations = pd.read_csv(csv_file)
        # initialising the root directory
        self.root_dir = root_dir
        # initialising transforms
        self.transforms = transforms

    # testing function to return the amount of labels read
    def __len__(self):
        return(len(self.annotations))

    # function to return an image and its label
    def __getitem__(self, index):
        # find the image path in the list
        img_path = self.annotations.iloc[index, 1]
        # if it is a matlab file then use the matlab reader
        if '.mat' in str(img_path):
            img = returnImage(img_path)
        # otherwise use pillow to open the image
        elif '.png' in str(img_path):
            img = Image.open(str(img_path).lstrip(
                "b'").rstrip("'")).convert(mode='I')
        # if the image is not standard size then resize it
        if img.size != (512, 512):
            img = img.resize((512, 512))

        # convert the image into an array and then a numpy array of data type float
        image = asarray(img).astype(np.float)

        # find the corresponding label
        label = torch.tensor(int(self.annotations.iloc[index, 0])) - 1

        # transform
        if self.transforms:
            image = self.transforms(image)

        # return the image and label
        return(image, label)
