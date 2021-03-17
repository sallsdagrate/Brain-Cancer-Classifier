# import matlabReader
# import normaliseNo
# import findPaths
# import randomSelector
from numpy import genfromtxt

from dataLoaderFile import matlabReader
from dataLoaderFile import normaliseNo
from dataLoaderFile import findPaths
from dataLoaderFile import randomSelector

# from . import matlabReader

# normalise
# find paths of all images
# randomly select images
# load dataList into program
# split into batches ready for training


def loadLists():
    # read csv file into numpy array
    data = genfromtxt(
        'classifier/dataLoaderFile/NEA_data/extracted/randomPaths.csv', delimiter=',', dtype=None)
    # print(data)
    return data


def splitLists(data):
    # input batch size
    # repeat until batch size is a factor or data list size
    # fact = False
    # while fact == False:
    # batchSize = 0
    # while batchSize < 1:
    #     batchSize = int(input('enter batch size: '))
    # print(len(data) % batchSize)

    # if len(data) % batchSize == 0:
    #     print('true')
    #     fact = True
    batchSize = 20
    numOfBatches = len(data)/batchSize  # save number of batches
    # print(numOfBatches)
    # split data into batches
    batches = [data[batchSize*i:batchSize*(i+1)]
               for i in range(int(numOfBatches))]
    # for n in range(int(numOfBatches)):
    #     print(batches[n]) #print batches
    # output for testing
    print(f'{numOfBatches} batches, {batchSize} batchsize')
    return (batches)


def main():
    # normaliseNo.main()
    # findPaths.main()
    # print('paths')
    # randomSelector.main()
    # print('selector')
    data = loadLists()
    return(splitLists(data))


# data = main()
# print(data)


def test():
    print('this is a test')
