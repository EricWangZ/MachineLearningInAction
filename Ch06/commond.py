#coding=utf-8

import os
os.chdir('C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch06')

from numpy import *

# 第六章 支持向量机 （SVM，Support Vector Machines）
# 部分认为， SVM是最好的现成的分类器。 SVM有很多实现，这里使用“序列最小优化” （SMO， Sequential Minimal Optimization)

# 6.1 基于最大间隔分隔数据


# 6.3.2 应用简化版SMO算法处理小规模数据集

import svmMLiA
dataArr,labelArr = svmMLiA.loadDataSet('testSet.txt')
labelArr

b,alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)

print b
print alphas[alphas>0]

shape(alphas[alphas>0])

for i in range(100):
    if alphas[i]>0.0: print dataArr[i],labelArr[i]
    
dataArr,labelArr = svmMLiA.loadDataSet('testSet.txt')
b,alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
print b, alphas

ws = svmMLiA.calcWs(alphas, dataArr, labelArr)
print ws

datMat = mat(dataArr)
datMat[0] * mat(ws) + b 
print labelArr[0]

datMat[2]*mat(ws)+ b 
print labelArr[2]

datMat[1]*mat(ws)+ b 
print labelArr[1]
