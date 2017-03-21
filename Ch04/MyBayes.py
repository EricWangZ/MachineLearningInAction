'''
Created on March 20, 2017

@author: Eric
'''
from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please', 'my'],
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
#    p0Num = zeros(numWords); p1Num = zeros(numWords)  #array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.])    
    p0Num = ones(numWords); p1Num = ones(numWords)  #array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 1.,  1.,  1.,  1.,  1.,  1.]) 
#    p0Denom = 1.0; p1Denom = 1.0                    #change to 2.0
    p0Denom = 2.0; p1Denom = 2.0                    #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log(), 
    p0Vect = log(p0Num/p0Denom)         #change to log()
    return p0Vect,p1Vect,pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    # log( p(w1/c1) ) + log (p(w2/c1)) +...+ log (p(w32)/c1) + log ( c1)  ----> p(w1/c1)*p(w2/c1)*...*p(w32/c1)*p(c1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)   # log( p(w1/c0) ) + log (p(w2/c0)) +...+ log (p(w32)/c0) + log ( c0)  ----> p(w1/c0)*p(w2/c0)*...*p(w32/c0)*p(c0)
    if p1 > p0:                                           # did not  / p (w)
        return 1
    else: 
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()     # loaddata
    myVocabList = createVocabList(listOPosts)  # create vocabulary list
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))        # get trainMat matrics
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))         # get p(wi/ci), p(ci)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))            # array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)    

def bagOfWords2VecMN(vocabList, inputSet):     # similar as setOfWords2Vec, but return different
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec               # return [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 2]

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
    
def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)  # [['codeine', '15mg', 'for', '203', 'visa', 'only', 'codeine', 'methylmorphine', 'narcotic', 'opioid', 'pain', 'reliever', 'have', '15mg', '30mg', 'pills', '15mg', 'for', '203', '15mg', 'for', '385', '15mg', 'for', '562', 'visa', 'only']]
        fullText.extend(wordList)  # ['codeine', '15mg', 'for', '203', 'visa', 'only', 'codeine', 'methylmorphine', 'narcotic', 'opioid', 'pain', 'reliever', 'have', '15mg', '30mg', 'pills', '15mg', 'for', '203', '15mg', 'for', '385', '15mg', 'for', '562', 'visa', 'only']
        classList.append(1)    # [1]
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)   # [['codeine', '15mg', 'for', '203', 'visa', 'only', 'codeine', 'methylmorphine', 'narcotic', 'opioid', 'pain', 'reliever', 'have', '15mg', '30mg', 'pills', '15mg', 'for', '203', '15mg', 'for', '385', '15mg', 'for', '562', 'visa', 'only'], ['peter', 'with', 'jose', 'out', 'town', 'you', 'want', 'meet', 'once', 'while', 'keep', 'things', 'going', 'and', 'some', 'interesting', 'stuff', 'let', 'know', 'eugene']]
        fullText.extend(wordList)  # ['codeine', '15mg', 'for', '203', 'visa', 'only', 'codeine', 'methylmorphine', 'narcotic', 'opioid', 'pain', 'reliever', 'have', '15mg', '30mg', 'pills', '15mg', 'for', '203', '15mg', 'for', '385', '15mg', 'for', '562', 'visa', 'only', 'peter', 'with', 'jose', 'out', 'town', 'you', 'want', 'meet', 'once', 'while', 'keep', 'things', 'going', 'and', 'some', 'interesting', 'stuff', 'let', 'know', 'eugene']
        classList.append(0)   # [1, 0, 1, 0]
    vocabList = createVocabList(docList)   #create vocabulary ['588', 'and', 'major', 'methylmorphine', 'over', '570', 'mba', 'narcotic', 'creative', 'done', 'art', 'noprescription', 'fine', 'jose', 'management', 'yay', 'working', 'brained', 'interesting', 'top', 'strategy', 'doing', 'only', 'going', 'way', '15mg', 'eugene', '750', 'both', '562', 'express', 'delivery', 'watson', 'know', 'focusing', 'strategic', 'with', 'reliever', 'town', 'school', 'while', 'stuff', 'today', 'cards', 'meet', 'more', 'opioid', 'program', 'right', '199', 'have', '195', 'some', 'hydrocodone', 'design', 'want', 'peter', 'pills', 'out', 'for', 'check', 'things', 'fedex', 'vicodin', '325', 'new', 'you', 'approach', 'brand', '200', '203', 'pain', '30mg', 'free', 'visa', 'let', '385', '120', 'required', 'codeine', 'days', 'keep', 'credit', 'the', 'cca', 'order', 'once']
    trainingSet = range(50); testSet=[]           #create test set
    for i in range(10):                            # randomly select 10 documents as testing set, remove these 10 from training set
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:            # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))       # [[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 2, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]]
        trainClasses.append(classList[docIndex])       # [1, 1, 0, 0, 1, 1,....]
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        # classify the remaining items, and calculate the error rate
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)
    #return vocabList,fullText

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:100]       # original :30

def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):                                        # get one entry
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)                   # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)                   # SF is class 0
    vocabList = createVocabList(docList)                          #create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)                 #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]                      #create test set
    for i in range(20):                                           # select 20 testing set
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:                                  #train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:                                     #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]
