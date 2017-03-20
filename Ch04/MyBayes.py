'''
Created on March 20, 2017

@author: Eric
'''
from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
                 
def createVocabList(dataSet): # dataSet:  [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], ['stop', 'posting', 'stupid', 'worthless', 'garbage'], ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) ##union of the two sets; vocabSet:  set(['cute', 'love', 'help', 'I', 'park', 'is', 'problems', 'not', 'dalmation', 'flea', 'him', 'maybe', 'please', 'dog', 'to', 'stupid', 'so', 'take', 'has', 'my'])
    return list(vocabSet)   # list(vocabSet):  ['cute', 'love', 'help', 'I', 'park', 'is', 'problems', 'not', 'dalmation', 'flea', 'him', 'maybe', 'please', 'dog', 'to', 'stupid', 'so', 'take', 'has', 'my']

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)  # returnVec: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for word in inputSet:
        if word in vocabList:   # word: my;  vocabList: ['cute', 'love', 'help', 'garbage', 'quit', 'I', 'problems', 'is', 'park', 'stop', 'flea', 'dalmation', 'licks', 'food', 'not', 'him', 'buying', 'posting', 'has', 'worthless', 'ate', 'to', 'maybe', 'please', 'dog', 'how', 'stupid', 'so', 'take', 'mr', 'steak', 'my']
            returnVec[vocabList.index(word)] = 1    # returnVec: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec


def trainNB0(trainMatrix,trainCategory):    # trainCategory: [0, 1, 0, 1, 0, 1]
    numTrainDocs = len(trainMatrix)         # numTrainDocs: 6
    numWords = len(trainMatrix[0])          # numWords: 32
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)  #array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 1.,  1.,  1.,  1.,  1.,  1.]) 
    p0Denom = 2.0; p1Denom = 2.0                    #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)         #change to log()
    return p0Vect,p1Vect,pAbusive


