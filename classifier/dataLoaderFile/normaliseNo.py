from scipy.io import loadmat
from PIL import Image, ImageEnhance
import numpy as np
from numpy import savetxt
import os
import cv2
import h5py
# same imports

directory = 'classifier/dataLoaderFile/NEA_data/extracted/images'  # define directory


def main():
    n = 0  # set count for testing
    for filename in os.listdir(directory + '/no'):  # loop for all images
        # print ('ya')
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            n = n + 1
            filePath = directory + '/no/' + filename  # define filepath
            oldimage = Image.open(filePath)  # open image
            newimage = oldimage.resize((512, 512))  # resize image
            print(newimage.size)  # print image size for testing
            # save resized image with new name in new file
            newimage.save(directory + '/nomodified/' + str(n) + 'no.png')

# main()


def checkDims():
    for filename in os.listdir(directory + '/nomodified'):  # loop for all images
        filePath = directory + '/nomodified/' + filename  # define filepath
        oldimage = Image.open(filePath)
        if oldimage.size != (512, 512):
            print(filePath)


# checkDims()
