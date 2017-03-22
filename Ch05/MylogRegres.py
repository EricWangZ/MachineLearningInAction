
'''
Created on Mar 21, 2017
Logistic Regression Working Module
@author: Eric
'''

from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('2.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()   # '-0.017612\t14.053064\t0\n' --> '-0.017612\t14.053064\t0' --> ['-0.017612', '14.053064', '0']
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])   # [[1.0, -0.017612, 14.053064], [1.0, -1.395634, 4.662541]]
        labelMat.append(int(lineArr[2]))   #  [0, 1]
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))      # theta 
    for k in range(maxCycles):              #    grad = ( X' * (y - sigmoid(X*theta) ) ) / m ;    
        h = sigmoid(dataMatrix*weights)     #    (sigmoid(X*theta)
        error = (labelMat - h)              #    (y - sigmoid(X*theta) )
        weights = weights + alpha * dataMatrix.transpose()* error        # theta := theta + alpha * ( X' * (y - sigmoid(X*theta) ) )
    return weights


''' 
     weights, theta 
            array([[ 1.],
                   [ 1.],
                   [ 1.]])
dataMatrix
matrix([[  1.00000000e+00,  -1.76120000e-02,   1.40530640e+01],
        [  1.00000000e+00,  -1.39563400e+00,   4.66254100e+00],
        [  1.00000000e+00,  -7.52157000e-01,   6.53862000e+00],
        [  1.00000000e+00,  -1.32237100e+00,   7.15285300e+00],
        [  1.00000000e+00,   4.23363000e-01,   1.10546770e+01],
        ...
labelMat
matrix([[0],
        [1],
        [0],
        [0],
        [0],
        ...

'''

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]      # 0 = w0x0+w1x1+w2x2
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()


# update weights in every step
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]    # theta := theta + alpha * ( X' * (y - sigmoid(X*theta) ) )
    return weights

# improved stocGradAscent1
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #1st improvement: alpha decreases with iteration, but > 0
            randIndex = int(random.uniform(0,len(dataIndex)))   # 2nd improvement: randomly select one set of x, go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights




def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)     # from 0 to 20
        trainingLabels.append(float(currLine[21]))    # 21
        
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)   # calculate theta
    
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):    # forecast by using theta
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))