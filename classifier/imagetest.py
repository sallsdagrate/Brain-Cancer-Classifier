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
count = 0
allResults = open('classifier/dataLoaderFile/NEA_data/extracted/skimages/allResults.txt', 'a')
for x in unzipped[1]:
    
    print (str(x))

    if '.mat' in str(x):
        img = matlabReader.returnImage(x)
        # path = 'classifier/dataLoaderFile/NEA_data/extracted/skimages/' + str(x).lstrip("b'classifier/dataLoaderFile/NEA_data/extracted/").rstrip(".mat'") + '/'
        # filename = 'classifier/dataLoaderFile/NEA_data/extracted/skimages/' + str(x).lstrip("b'classifier/dataLoaderFile/NEA_data/extracted/").rstrip(".mat'") + ".jpg"
        count+=1
    elif '.png' in str(x):
        # path needs to be adjusted slightly to work and the image must be translated into image mode 'I' like the rest of them
        count+=1
        # path = 'classifier/dataLoaderFile/NEA_data/extracted/skimages/' + str(x).lstrip("b'classifier/dataLoaderFile/NEA_data/extracted/mages/nomodified/").rstrip(".png'") + '/'
        img = Image.open(str(x).lstrip("b'").rstrip("'")).convert(mode='I')
        # filename = 'classifier/dataLoaderFile/NEA_data/extracted/skimages/' + str(x).lstrip("b'classifier/dataLoaderFile/NEA_data/extracted/mages/nomodified/").rstrip(".png'") + ".jpg"
    
    path = 'classifier/dataLoaderFile/NEA_data/extracted/skimages/' + str(count) + '/'
    filename = path + str(count) + ".jpg"
    
    try:
        os.makedirs(path)
    except:
        next

    image = asarray(img)
    # edgefilename = filename + 'edges/'
    plt.imsave(filename, image)
    edges = filters.sobel(image)
    edgefilename = path + str(count) + 'edge.jpg'
    plt.imsave(edgefilename, edges)
    resultfile = open(path + str(count) + 'result.txt', 'a')
    if 'no' in str(x):
        result = 4.0
    else:
        result = matlabReader.returnLabel(x)
    resultfile.write(str(result))
    resultfile.close()
    allResults.write(str(f'{str(count)} : {result}\n'))
    # print(result)
    # if not('no' in str(x)):
        # borderfile = open(path + str(count) + 'border.txt', 'a')
        # border = matlabReader.returnBorder(x)
        # borderfile.write(str(border))
        # borderfile.close()
        # borderimg = matlabReader.returnBorder(x)
        # border = asarray(borderimg)
        # borderfilename = path + str(count) + 'border.jpg'
        # plt.imsave(borderfilename, border)
        # io.imshow(border)
        # io.show()
    # print(filename)
allResults.close()
print(count)

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