from PIL import Image, ImageEnhance
from numpy import asarray
from dataLoaderFile import matlabReader
import os
# img = matlabReader.returnImage('classifier/dataLoaderFile/NEA_data/extracted/300.mat')

import matplotlib.pyplot as plt

from skimage import data, io, filters

# image = asarray(img)
# edges = filters.sobel(image)
# io.imshow(edges)
# io.show()
# io.imshow(image)
# io.show()

# print(image)

# plt.imsave('skimage.png', image)

file = open(
        'classifier/dataLoaderFile/NEA_data/extracted/randomPaths.csv', 'a')

from dataLoaderFile import dataLoader

data = dataLoader.loadLists()
# print (data)
unzipped = [[i for i, j in data],
                [j for i, j in data]]
# print(unzipped[1])
# count = 0
for x in unzipped[1]:
    
    if '.mat' in str(x):
        img = matlabReader.returnImage(x)
        filename = 'classifier/dataLoaderFile/NEA_data/extracted/skimages/' + str(x).lstrip("b'classifier/dataLoaderFile/NEA_data/extracted/").rstrip(".mat'") + ".jpg"
    elif '.png' in str(x):
        # path needs to be adjusted slightly to work and the image must be translated into image mode 'I' like the rest of them
        img = Image.open(str(x).lstrip("b'").rstrip("'")).convert(mode='I')
        filename = 'classifier/dataLoaderFile/NEA_data/extracted/skimages/' + str(x).lstrip("b'classifier/dataLoaderFile/NEA_data/extracted/mages/nomodified/").rstrip(".png'") + ".jpg"
    
    image = asarray(img)
    edges = filters.sobel(image)
    edgefilename = filename + 'edges/'
    plt.imsave(filename, image)
    plt.imsave(edgefilename, edges)
    # count+=1
    print(x)

# print(count)

# pixels = list(img.getdata())
# print(pixels)
# width, height = img.size
# pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
# print(pixels)

# import cnn

# tensor = cnn.trans(img)
# print(tensor)
# img.save('testimage.png')

# newimg = Image.open('classifier/testimage.png')
# newimg.show()

# print(img.mode)
# # img.show()
# img = img.convert(mode = 'RGB')
# print(img.mode)
# img.show()

# enhancer = ImageEnhance.Contrast(img)
# brightness = ImageEnhance.Brightness(img)
# imout = enhancer.enhance(0.5)
# imout.show()
# bimout = brightness.enhance(0.5)
# bimout.show()