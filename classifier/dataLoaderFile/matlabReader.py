from scipy.io import loadmat
from PIL import Image, ImageEnhance
import numpy as np
from numpy import savetxt
import os
import cv2
import h5py


directory = 'dataLoaderFile/NEA_data/extracted/'  # define location of data
n = 0  # set count to 0 for testing

# function to return image from filePath


def returnImage(filePath):
    readFile = h5py.File(filePath, 'r')  # open in read mode
    fileData = readFile['cjdata']
    # find image field and convert to numpy array
    fileScan = np.array(fileData['image'])
    imageOut = Image.fromarray(fileScan)  # convert array into image
    # print(imageOut)
    return (imageOut)


def extractImage(filename, filePath):
    # changing name of image to png
    imageName = filename.replace('.mat', 'img.png')
    # joining path and name for new location
    imagePath = os.path.join(directory + 'images/', imageName)

    imageOut = returnImage(filePath)  # extract image
    # save image with new name and location
    imageOut.save(filename.replace('.mat', 'img.png'))

    print(imagePath)


def returnLabel(filePath):
    readFile = h5py.File(filePath, 'r')  # open in read mode
    fileData = readFile['cjdata']

    newLabel = fileData['label']
    return(np.array(newLabel[0][0]))


def extractLabel(filename, filePath):
    labelsFile = open(directory + "labels/labelsfile.txt", "a")

    val = returnLabel(filename, filePath)  # stores the real label value
    print(val)
    # count[int(val)] = count[int(val)] + 1
    labelsFile.write(str(val) + "\n")  # writes new label on new line

    labelsFile.close()  # closes file


def returnBorder(filePath):
    readFile = h5py.File(filePath, 'r')  # open in read mode
    fileData = readFile['cjdata']
    newBorder = np.array(fileData['tumorBorder'])
    imageOut = Image.fromarray(newBorder)  # convert array into image
    return(imageOut)


def extractBorder(filename, filePath):
    imageOut = returnBorder(filename, filePath)
    print(imageOut)
    savetxt(directory + 'borders/' + filename.replace('.mat', '') +
            'border' + '.csv', imageOut, delimiter=',')


def returnMask(filePath):
    readFile = h5py.File(filePath, 'r')  # open in read mode
    fileData = readFile['cjdata']
    newMask = np.array(fileData['tumorMask'])
    imageOut = Image.fromarray(newMask)  # convert array into image
    return(imageOut)


def extractMask(filename, filePath):
    imageOut = returnMask(filename, filePath)
    print(imageOut)
    # changing name of image to png
    imageName = filename.replace('.mat', 'mask.png')
    # joining path and name for new location
    imagePath = os.path.join(directory + 'masks/', imageName)
    imageOut.save(imagePath)

# count = [0, 0, 0, 0]


def main():
    # looping for every file path in directory
    for filename in os.listdir(directory):
        if filename.endswith('.mat'):  # only looking at matlab files
            filePath = directory + filename  # defining raw path
            print(filename, filePath)

            extractImage(filename, filePath)
            extractLabel(filename, filePath)
            extractBorder(filename, filePath)
            extractMask(filename, filePath)

            # stop at 25 for testing
            n = n+1
            print(n)
            if n >= 25:
                break

# main()

# print (count)
