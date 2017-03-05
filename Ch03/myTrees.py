'''
Created on Mar 05, 2017
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Eric Wang
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):                # calculate the Shannon value for one dataSet, only use the labels
    numEntries = len(dataSet)               # get the length of the dataSet
    labelCounts = {}                        # create a dictionary, get the number of unique elements and their occurance, 
    for featVec in dataSet:                 # get one row each time from dataSet, 
        currentLabel = featVec[-1]          # get the element of the last column,  always the last column as label!!!
        
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0    # if currentLabel not in the dictionary, initial this in dictionary as 0
        labelCounts[currentLabel] += 1      # currentlabel in dictionary + 1
    
    shannonEnt = 0.0
    for key in labelCounts:                 # go through the keys (yes, no, maybe) in the labelcounts
        prob = float(labelCounts[key])/numEntries       # cal probability of this key in the whole dataSet
        shannonEnt -= prob * log(prob,2)                # calculate the shannon value: H = - sum(p(xi)*log2 p(xi))
    return shannonEnt


def splitDataSet(dataSet, axis, value):         # split data based on the feature, dataSet, axis(index of the feature to split), value(the value to be selected)
    retDataSet = []                             # create a new list, as python pass the reference, any change is global
    for featVec in dataSet:                     # get each row of the dataSet
        if featVec[axis] == value:              # if the element(feature) in this row match value
            reducedFeatVec = featVec[:axis]         # choose the elements before axis, [0, 1, yes], take [0]
            reducedFeatVec.extend(featVec[axis+1:]) # extended with the elements after axis, take [yes] so to chop out axis used for splitting, get [0, yes]
            retDataSet.append(reducedFeatVec)       # add the selected row into the new list
    return retDataSet                               # return the new list
    
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1           # get the length of row, the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)       # calculate the shannon with the current set
    bestInfoGain = 0.0; bestFeature = -1        
    for i in range(numFeatures):                # iterate over all the features, 
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature:i [1,1,1,0,0], get that column
        uniqueVals = set(featList)                      # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:                        # for each unique value of that feature, iteration
            subDataSet = splitDataSet(dataSet, i, value)    # split data, dataSet, axis = i (the current feature), value = unique value
            prob = len(subDataSet)/float(len(dataSet))      # calculate the % of this subdataset in the whole dataset
            newEntropy += prob * calcShannonEnt(subDataSet)     # calculate the p * shannon value (of this subdataSet),  sum up to get the whole shannon value for the feature
        infoGain = baseEntropy - newEntropy         # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):               # compare this to the best gain so far
            bestInfoGain = infoGain                 # if better than current best, set to best
            bestFeature = i                         # mark this feature
    return bestFeature                              # returns the best feature
