
#coding=utf-8

from sys import path
path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch13')
#path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch13')

from numpy import *
import myPCA

# 第四部分 其他工具
# 第十三章 利用PCA来简化数据

# 13.1 降维技术
# PCA分析； 因子分析； 独立成分分析

# 13.2 PCA分析： Principle components Analysis
# 13.1.1 移动坐标轴
# 坐标轴旋转以覆盖更多点
# 求特征值、特征向量  AV = lamda * V

# 13.2.2 在NumPy中实现PCA

dataMat = myPCA.loadDataSet('testSet.txt')
dataMat = myPCA.loadDataSet('testSet3.txt')
lowDMat, reconMat = myPCA.pca(dataMat,1)
# lowDMat, reconMat = myPCA.pca(dataMat,2)

print shape(lowDMat)

import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker = '^', s=90)
ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker = 'o', s=90, c='red')

plt.show()

# 13.3 示例： 利用PCA对半导体制造数据降维

dataMat = myPCA.replaceNanWithMean()
meanVals = mean (dataMat, axis=0)
meanRemoved = dataMat - meanVals
covMat = cov (meanRemoved, rowvar = 0)
eigVals,eigVects = linalg.eig(mat(covMat))

print eigVals
# 前15个特征大于10**5





