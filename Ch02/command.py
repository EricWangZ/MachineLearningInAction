
from sys import path
# path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\MachineLearningInAction\Ch02')
path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch02')


import myKNN
dataSet,labels = myKNN.createDataSet()
myKNN.classify0([0,0],dataSet,labels,3)

datingDataMat, datingLabels = myKNN.file2matrix('datingTestSet2.txt')

'''
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
plt.show()


ax.scatter(datingDataMat[:,1], datingDataMat[:,0], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()

'''

normMat, ranges, minVals = myKNN.autoNorm(datingDataMat)
# print normMat, ranges,minVals

reload(myKNN)
'''
myKNN.datingClassTest()

myKNN.classifyPerson()

'''
myKNN.img2vector('testDigits/0_13.txt')

myKNN.handwritingClassTest()