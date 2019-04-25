import math
import numpy as np
from collections import Counter



def loadTrainingData(trainingFile, testingFile, validationFile, trainingSet=[], testSet=[], validationSet=[]): 
    trainingSetFile = open (trainingFile, "r")
    traininglines = trainingSetFile.readlines()
    trainingSetFile.close()
    for unFormattedLine in traininglines: 
        line = unFormattedLine.split()
        currentLine = []
        for char in line:
            currentLine.append(int(char))
        trainingSet.append(currentLine)

    testingSetFile = open (testingFile, "r")
    testingLines = testingSetFile.readlines()
    testingSetFile.close()
    for unFormattedLine in testingLines: 
        line = unFormattedLine.split()
        currentLine = []
        for char in line:
            currentLine.append(int(char))        
        testSet.append(currentLine)

    validationSetFile = open (validationFile, "r")
    validationLines = validationSetFile.readlines()
    validationSetFile.close()
    for unFormattedLine in validationLines: 
        line = unFormattedLine.split()
        currentLine = []
        for char in line:
            currentLine.append(int(char))
        validationSet.append(currentLine)

#returns the accuracy
def knn(k, testPoints, trainingSet):

    predictions = []
    for point in testPoints:
        prediction = predict(k, point, trainingSet)
        predictions.append(prediction)
    
    return predictions

def getAccuracy(predictions, testPoints):
    i = 0
    numCorrect = 0
    for prediction in predictions:
        print(str(prediction) + ' : ' + str(testPoints[i][-1]))
        if int(prediction) == int(testPoints[i][-1]):
            numCorrect += 1
            print('yer mams a  hoe' + str(numCorrect))

        i += 1
    return numCorrect/len(testPoints)

#outputs the most common of the neighbors in the neigborhood of k 
def predict(k, test, trainingSet): 
    distances = []
    neighborhood = []

    i = 0
    for trainingPoint in trainingSet:
        distances.append([getDistance(trainingPoint, test), i])
        i += 1
        
    
    distances = sorted(distances)

    for i in range(k):
        neighborhood.append(trainingSet[distances[i][1]][-1])

    data = Counter(neighborhood)

    print(str(neighborhood) + ' : ' + str(max(neighborhood, key=data.get)))
    return max(neighborhood, key=data.get)





def getDistance(first, second): 
    firstList = np.array(first[:-1])
    secondList =  np.array(second[:-1])

    dist = np.linalg.norm(firstList-secondList)

    return dist

def testLoadTrainingData(filename, trainingSet=[], testSet=[], validationSet=[]):
    trainingSetFile = open (filename, "r")
    trainingline = trainingSetFile.readline()
    trainingSetFile.close()
    current = []
    for char in trainingline:
        current.append(char)


    
trainingSet = []
testSet = []
validationSet = []

#testLoadTrainingData('pa1train.txt', trainingSet, testSet, validationSet)

loadTrainingData('pa1train.txt', 'pa1train.txt', 'pa1validate.txt', trainingSet, testSet , validationSet)
predictions = knn(3, testSet, trainingSet)
accuracy = getAccuracy(predictions, testSet)
print(accuracy) 


x = predict(3, testSet[700], trainingSet)
print(x)