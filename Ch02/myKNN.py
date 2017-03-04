'''
@author: EricWang

from sys import path
path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\MachineLearningInAction\Ch02')
path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch02')
'''

from numpy import *
import operator
from os import listdir


def createDataSet():
    dataSet = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return dataSet, labels

#dataSet = array([[1.0,1.1, 'A', 3],[1.0,1.0, 'A',4],[0,0, 'B', 1],[0,0.1, 'B',2]])
#

def classify0(inX, dataSet, labels, k):

#    x1 = column_stack((dataSet, labels))
    dataSetSize = dataSet.shape[0]                  # get dataSet length
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  # extend inX to the length of dataSet, and - dataSet
    sqDiffMat = diffMat**2                          # cal ^2 ---  (xA0 - xB0)^2
    sqDistances = sqDiffMat.sum(axis=1)             # cal +  ---   (xA0 - xB0)^2+(xA1 - xB1)^2     
    distances = sqDistances**0.5                    # cal ^0.5 
    sortedDistIndicies = distances.argsort()        # get the index of sorted distance
    classCount={}          
    for i in range(k):                              # for i -> k
        voteIlabel = labels[sortedDistIndicies[i]]  # get the label based on the sorted index
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1   # +1 for the labels
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) # get the top label
    return sortedClassCount[0][0]                   # return the top

def file2matrix(filename):
    fr = open(filename)                         # open the file
    numberOfLines = len(fr.readlines())         # get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        # prepare matrix to return
    classLabelVector = []                       # prepare labels return   
    fr = open(filename)                         # open the file again
    index = 0       
    for line in fr.readlines():                 # go through all lines
        line = line.strip()                     # 
        listFromLine = line.split('\t')         #convert string 0.1 0.2 0.3 3 to [0.1,0.2,0.3,3]
        returnMat[index,:] = listFromLine[0:3]  # get [0.1,0.2,0.3]
        classLabelVector.append(int(listFromLine[-1])) # get [3]
        # print index,returnMat[index,:], classLabelVector[index]
        index += 1
    return returnMat,classLabelVector
    
def autoNorm(dataSet):                              # normalize = (old value - min) / (max - min)
    minVals = dataSet.min(0)                        # get min value for each column
    maxVals = dataSet.max(0)                        # get max value for each column
    ranges = maxVals - minVals                      # range = max - min
    normDataSet = zeros(shape(dataSet))             # initial a new dataSet
    m = dataSet.shape[0]                            # get the length of dataset
    normDataSet = dataSet - tile(minVals, (m,1))    # dataSet - Min value set  --- element wise substruct
    normDataSet = normDataSet/tile(ranges, (m,1))   # (dataSet - Min) / (max - min)element wise divide
    return normDataSet, ranges, minVals
   
def datingClassTest():
    hoRatio = 0.10      # the first 10% of data as test data, the rest as the normal dataSet
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')      #load data set from file
    normMat, ranges, minVals = autoNorm(datingDataMat)                  #normalize the dataSet
    m = normMat.shape[0]                        # get the dataSet length
    numTestVecs = int(m*hoRatio)                # get the top 10% length, say 0.1 * 1000 = 100
    errorCount = 0.0
    for i in range(numTestVecs):                # from 0 to 99
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],10)
                                                # classify the ith data, the normal dataSet is from 100 to 999
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0         # if classification != label, error +1
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount

def classifyPerson():
    resultList = ['Not at all','in small doses', 'in large doses']              # define return result strings
    percentTats = float (raw_input("percentage of time spent playing video games?"))    # get parameter 1
    ffMiles = float (raw_input("frequent flier miles earned per year?"))                # get parameter 2
    iceCream = float (raw_input("liters of ice cream consumed per year?"))              # get parameter 3
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')                     # get dataSet, Labels
    normMat, ranges, minVals = autoNorm(datingDataMat)                                  # normalize dataSet
    inArr = array([ffMiles, percentTats, iceCream])                                     # form parameters to one array
    classifierResult = classify0((inArr-minVals)/ranges, normMat,datingLabels,3)        # classify
    print "You will probably like his person: ", resultList[classifierResult - 1]       # return the string, index from 0

def img2vector(filename):
    returnVect = zeros((1,1024))        # initial vector
    fr = open(filename)                 # get file
    for i in range(32):                 # from 1st line to 32th line
        lineStr = fr.readline()         # get the line
        for j in range(32):             # get the int one-by-one
            returnVect[0,32*i+j] = int(lineStr[j])  # copy the int one-by-one, one row vector
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')            # get the training dir, load the training set
    m = len(trainingFileList)                               # get the total file number
    trainingMat = zeros((m,1024))                           # training set, row = file number, column = 32*32
    for i in range(m):                                      # go through all training file, to set dataSet, Labels
        fileNameStr = trainingFileList[i]                   # get ith file name
        fileStr = fileNameStr.split('.')[0]                 # take off .txt, only get the file name
        classNumStr = int(fileStr.split('_')[0])            # take number
        hwLabels.append(classNumStr)                        # put number into labels
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)    # set the image to the dataSet
    testFileList = listdir('testDigits')                    # get the test filr dir, iterate through the test set
    errorCount = 0.0                        
    mTest = len(testFileList)                               # the total test file number
    for i in range(mTest):                                  # go through all testing files
        fileNameStr = testFileList[i]                       # get the file name
        fileStr = fileNameStr.split('.')[0]                 # take off .txt
        classNumStr = int(fileStr.split('_')[0])            # get the labels
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr) # get the test data
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3) # get the classification
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0     # compare the classificaiton with label, count total errors
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))   # count the % error rate

    
