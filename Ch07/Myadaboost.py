'''
Created on Mar 23, 2017
Adaboost is short for Adaptive Boosting
@author: Eric
'''
from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):   #just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    

def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf                          # init error sum, to +infinity
    for i in range(n):                      # loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();  # column(feature) i, min and max
        stepSize = (rangeMax-rangeMin)/numSteps         #  step size
        for j in range(-1,int(numSteps)+1):             #  loop over all range in current dimension
            for inequal in ['lt', 'gt']:                #  go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)  # prediction by calling stump classify with i, j, lessThan or morethan
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  # calc total error multiplied by D
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):   # full adaBoost
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)   #build Stump
        #print "D:",D.T
        
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))  # calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)                   # put each classifier into the array one-by-one, store Stump Params in Array
        #print "classEst: ",classEst.T
        
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) # exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              # Calc New D for next iteration
        D = D/D.sum()
        
        # calculate the result of the aggregate estimation
        aggClassEst += alpha*classEst
        #print "aggClassEst: ",aggClassEst.T         # final clarified results
        
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        #print "total error: ",errorRate
        if errorRate == 0.0: break

    return weakClassArr,aggClassEst


def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)    # do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):  # use the three classifier continuously
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'], \
                                 classifierArr[i]['thresh'], \
                                 classifierArr[i]['ineq'])    # call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        # print aggClassEst
    return sign(aggClassEst)

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat



