import os
import h5py
import numpy as np
from matlabReader import returnLabel  # reusing the same function to find label

# from dataLoaderFile.matlabReader import returnLabel

directory = 'classifier/dataLoaderFile/NEA_data/extracted/'  # define directory


def main():
    # open all files in append mode
    yes1PathsFile = open(directory + 'yes1PathsFile.txt', 'a')
    yes2PathsFile = open(directory + 'yes2PathsFile.txt', 'a')
    yes3PathsFile = open(directory + 'yes3PathsFile.txt', 'a')
    noPathsFile = open(directory + 'noPathsFile.txt', 'a')

    # loop for all matlab files
    for fileName in os.listdir(directory):
        if fileName.endswith('.mat'):
            path = directory + fileName

    # save path in appropriate file
            label = returnLabel(fileName, path)
            if label == 1:
                yes1PathsFile.write(path + '\n')
            elif label == 2:
                yes2PathsFile.write(path + '\n')
            elif label == 3:
                yes3PathsFile.write(path + '\n')

    # same for no
    for fileName in os.listdir(directory + 'images/nomodified'):
        if fileName.endswith('.png'):
            path = directory + 'images/nomodified/' + fileName
            noPathsFile.write(path + '\n')

    # close all files
    yes1PathsFile.close()
    yes2PathsFile.close()
    yes3PathsFile.close()
    noPathsFile.close()


# main()
