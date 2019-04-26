import math
from collections import Counter
import copy
import numpy as np


def loadTrainingData(trainingFile, testingFile, validationFile, projectionFile, trainingSet=[], testSet=[], validationSet=[], projectionSet=[], trainingSetLabel=[], testSetLabel=[], validationSetLabel=[]):
    trainingSetFile = open(trainingFile, "r")
    traininglines = trainingSetFile.readlines()
    trainingSetFile.close()
    for unFormattedLine in traininglines:
        line = unFormattedLine.split()
        currentLine = []
        for char in line[:-1]:
            currentLine.append(int(char))
        trainingSet.append(currentLine)
        trainingSetLabel.append(line[len(line)-1])

    projectionSetFile = open(projectionFile, "r")
    projectionlines = projectionSetFile.readlines()
    projectionSetFile.close()
    for unFormattedLine in projectionlines:
        line = unFormattedLine.split()
        currentLine = []
        for char in line:
            currentLine.append(float(char))
        projectionSet.append(currentLine)

    testingSetFile = open(testingFile, "r")
    testingLines = testingSetFile.readlines()
    testingSetFile.close()
    for unFormattedLine in testingLines:
        line = unFormattedLine.split()
        currentLine = []
        for char in line[:-1]:
            currentLine.append(int(char))
        testSet.append(currentLine)
        testSetLabel.append(line[len(line)-1])

    validationSetFile = open(validationFile, "r")
    validationLines = validationSetFile.readlines()
    validationSetFile.close()
    for unFormattedLine in validationLines:
        line = unFormattedLine.split()
        currentLine = []
        for char in line[:-1]:
            currentLine.append(int(char))
        validationSet.append(currentLine)
        validationSetLabel.append(line[len(line)-1])


def runKnn(k, testSet, trainingSet, testSetLabel, trainingSetLabel):
    predictions = knn(k, testSet, trainingSet, trainingSetLabel)
    accuracy = getAccuracy(predictions, testSetLabel)
    print(str(k) + ' : ' + str(accuracy))

    return accuracy


def knn(k, testPoints, trainingSet, trainingSetLabel):

    predictions = []
    for point in testPoints:
        prediction = predict(k, point, trainingSet, trainingSetLabel)
        predictions.append(prediction)

    return predictions


# outputs the most common of the neighbors in the neigborhood of k


def predict(k, test, trainingSet, trainingSetLabel):
    distances = []
    neighborhood = []

    i = 0
    for trainingPoint in trainingSet:
        distances.append([getDistance(trainingPoint, test), i])
        i += 1

    distances = sorted(distances)

    for i in range(k):
        neighborhood.append(trainingSetLabel[distances[i][1]])

    data = Counter(neighborhood)

    #print(str(neighborhood) + ' : ' + str(max(neighborhood, key=data.get)))

    return max(neighborhood, key=data.get)


def getDistance(first, second):
    firstList = np.array(first)
    secondList = np.array(second)

    dist = np.linalg.norm(firstList-secondList)

    return dist

# returns the accuracy
def getAccuracy(predictions, testSetLabel):
    i = 0
    numCorrect = 0
    for prediction in predictions:
        if int(prediction) == int(testSetLabel[i]):
            #print("ur mam's a hoe " + str(numCorrect))
            numCorrect += 1
        i += 1
    return (1 - float(numCorrect)/float(len(testSetLabel)))


def project(subject, target):
    return np.matmul(subject, target)


trainingSet = []
testSet = []
validationSet = []
trainingSetLabel = []

testSetLabel = []
validationSetLabel = []
projectionSet = []

newTrainingSet = []
newTestSet = []
newValidationSet = []


loadTrainingData('pa1train.txt', 'pa1train.txt', 'pa1validate.txt',
                 'projection.txt', trainingSet, testSet, validationSet, projectionSet, trainingSetLabel, testSetLabel, validationSetLabel)

print("Computing training error on the original dataset")
oneTrainingOriginal = runKnn(1, trainingSet, trainingSet, trainingSetLabel, trainingSetLabel)
threeTrainingOrignal = runKnn(3, trainingSet, trainingSet, trainingSetLabel, trainingSetLabel)
fiveTrainingOringal = runKnn(5, trainingSet, trainingSet, trainingSetLabel, trainingSetLabel)
nineTrainingOriginal = runKnn(9, trainingSet, trainingSet, trainingSetLabel, trainingSetLabel)
fifteenTrainingOriginal = runKnn(15, trainingSet, trainingSet, trainingSetLabel, trainingSetLabel)

print("computing the validation error on the orignal dataset")
validationOriginal = []
oneValidationOriginal = runKnn(1, validationSet, trainingSet, validationSetLabel, trainingSetLabel)
threeValidationOriginal = runKnn(3, validationSet, trainingSet, validationSetLabel, trainingSetLabel)
fiveValidationOriginal = runKnn(5, validationSet, trainingSet, validationSetLabel, trainingSetLabel)
nineValidationOriginal = runKnn(9, validationSet, trainingSet, validationSetLabel, trainingSetLabel)
fifteenValidationOriginal = runKnn(15, validationSet, trainingSet, validationSetLabel, trainingSetLabel)

validationOriginal.append([oneValidationOriginal, 1])
validationOriginal.append([fiveValidationOriginal, 5])
validationOriginal.append([nineValidationOriginal, 9]) 
validationOriginal.append([fifteenValidationOriginal, 15])
originalMin = min(validationOriginal)
print('lowest error' + str(originalMin))


testErrorOriginal = runKnn(originalMin[1], testSet, trainingSet, testSetLabel, trainingSetLabel)

#Part 2
print("creating the new projected data")
tcopy = copy.deepcopy(trainingSet)
newTrainingSet = project(tcopy, projectionSet)

zcopy = copy.deepcopy(testSet)
newTestSet = project(zcopy, projectionSet)

ccopy = copy.deepcopy(validationSet)
newValidationSet = project(ccopy, projectionSet)

print("Computing the training error on the projected dataset")
oneTrainingProjected = runKnn(1, newTrainingSet, newTrainingSet, trainingSetLabel, trainingSetLabel)
threeTrainingProjected = runKnn(3, newTrainingSet, newTrainingSet, trainingSetLabel, trainingSetLabel)
fiveTrainingProjected = runKnn(5, newTrainingSet, newTrainingSet, trainingSetLabel, trainingSetLabel)
nineTrainingProjected = runKnn(9, newTrainingSet, newTrainingSet, trainingSetLabel, trainingSetLabel)
fifteenTrainingProjected = runKnn(15, newTrainingSet, newTrainingSet, trainingSetLabel, trainingSetLabel)

print("Computing the validaiton error on the projected dataset")
validationProjected = []
oneValidationProjected = runKnn(1, newValidationSet, newTrainingSet, trainingSetLabel, trainingSetLabel)
threeValidationProjected = runKnn(3, newValidationSet, newTrainingSet, trainingSetLabel, trainingSetLabel)
fiveValidationProjected = runKnn(5, newValidationSet, newTrainingSet, trainingSetLabel, trainingSetLabel)
nineValidationProjected = runKnn(9, newValidationSet, newTrainingSet, trainingSetLabel, trainingSetLabel)
fifteenValidationProjected = runKnn(15, newValidationSet, newTrainingSet, trainingSetLabel, trainingSetLabel)

validationProjected.append([oneValidationProjected, 1])
validationProjected.append([fiveValidationProjected, 5])
validationProjected.append([nineValidationProjected, 9])
validationProjected.append([fifteenValidationProjected, 15])

projectedMin = min(validationProjected)
print('lowest error' + str(projectedMin))

testErrorProjected = runKnn(projectedMin[1], newTestSet, trainingSet, testSetLabel, trainingSetLabel)