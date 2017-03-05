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
        currentLabel = featVec[-1]          # get the element of the last column,  which is the decision!
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
        featList = [example[i] for example in dataSet]  # create a list of all the values of this feature:i [1,1,1,0,0], get that column
        uniqueVals = set(featList)                      # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:                        # for each unique value of that feature, iteration -- splitDataSet -> calShannon -> add up
            subDataSet = splitDataSet(dataSet, i, value)    # split data, dataSet, axis = i (the current feature), value = unique value
            prob = len(subDataSet)/float(len(dataSet))      # calculate the % of this subdataset in the whole dataset
            newEntropy += prob * calcShannonEnt(subDataSet)     # calculate the p * shannon value (of this subdataSet),  sum up to get the whole shannon value for the feature
        infoGain = baseEntropy - newEntropy         # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):               # compare this to the best gain so far
            bestInfoGain = infoGain                 # if better than current best, set to best
            bestFeature = i                         # mark this feature
    return bestFeature                              # returns the best feature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]                # get the decision column, ['yes', 'yes', 'no', 'no', 'no']
    if classList.count(classList[0]) == len(classList):             # count how many 'yes', if all are 'yes', done
        return classList[0]     # return 'yes', #stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:    # get the length of the row, if 1, then all features used up. Stop splitting when there are no more features in dataSet
        return majorityCnt(classList)   # return the majority of the decision
    bestFeat = chooseBestFeatureToSplit(dataSet)    # choose the best feature
    bestFeatLabel = labels[bestFeat]                # get the label of the best feature, like 'no surfacing'
    myTree = {bestFeatLabel:{}}                     # Dictionary, myTree = {'no surfacing': {}} 
    del(labels[bestFeat])                           # delete the label of the best feature
    featValues = [example[bestFeat] for example in dataSet]     # get the value(column) of the best feature, [1, 1, 1, 0, 0]
    uniqueVals = set(featValues)                    # get the unique values in the best feature, set([0,1])
    for value in uniqueVals:                        # iterate set([0,1])
        subLabels = labels[:]                       # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels) # splitdata chop off the best feature values, subLabel already removed bestfeature
    return myTree                            


def classify(inputTree,featLabels,testVec):         # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}; ['no surfacing', 'flippers'];  [1, 0] 
    firstStr = inputTree.keys()[0]                  # 'no surfacing'              
    secondDict = inputTree[firstStr]                # {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}
    featIndex = featLabels.index(firstStr)          # featLabels.index('no surfacing') = 0;  ['no surfacing', 'flippers']
    key = testVec[featIndex]                        # 1 from [1, 0]
    valueOfFeat = secondDict[key]                   # {'flippers': {0: 'no', 1: 'yes'}}           
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):                  # store tree into txt
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):                             # retrieve from txt
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    



