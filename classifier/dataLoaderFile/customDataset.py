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


class dataset (Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return(len(self.annotations))

    def __getitem__(self, index):
        img_path = self.annotations.iloc[index, 1]
        if '.mat' in str(img_path):
            img = returnImage(img_path)
        elif '.png' in str(img_path):
            img = Image.open(str(img_path).lstrip(
                "b'").rstrip("'")).convert(mode='I')
        if img.size != (512, 512):
            img = img.resize((512, 512))
        image = asarray(img).astype(np.float)

        label = torch.tensor(int(self.annotations.iloc[index, 0])) - 1

        if self.transforms:
            image = self.transforms(image)

        return(image, label)
