import random

# function to randomly select a path in a file


def randomLine(path):
    lines = open(path).read().splitlines()
    return random.choice(lines)


def main():
    # create a new csv file
    file = open(
        'classifier/dataLoaderFile/NEA_data/extracted/randomPaths.csv', 'a')
    for n in range(100):  # select 100 paths for testing
        # randomly select a number between 1 and 4
        randint = random.randrange(1, 5)
        print(randint)
        # each random number result corresponds to a path file
        # every time a number is selected, a new path is found and stored
        if randint == 1:
            line = randomLine(
                'classifier/dataLoaderFile/NEA_data/extracted/yes1PathsFile.txt')
            file.write(str(randint) + ',' + line + '\n')
        elif randint == 2:
            line = randomLine(
                'classifier/dataLoaderFile/NEA_data/extracted/yes2PathsFile.txt')
            file.write(str(randint) + ',' + line + '\n')
        elif randint == 3:
            line = randomLine(
                'classifier/dataLoaderFile/NEA_data/extracted/yes3PathsFile.txt')
            file.write(str(randint) + ',' + line + '\n')
        elif randint == 4:
            line = randomLine(
                'classifier/dataLoaderFile/NEA_data/extracted/noPathsFile.txt')
            file.write(str(randint) + ',' + line + '\n')
    file.close()  # close file

# main()
