
#coding=utf-8

from sys import path
#path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch08')
path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch08')

from numpy import *
import Myregression
import matplotlib.pyplot as plt

# 第八章 利用回归预测数值型数据

# 用线性回归找到最佳拟合直线

xArr, yArr = Myregression.loadDataSet("ex0.txt")
ws = Myregression.standRegres(xArr, yArr)
print ws

xMat = mat(xArr);yMat = mat(yArr); yHat = xMat * ws

'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])

xCopy = xMat.copy(); xCopy.sort(0); yHat=xCopy*ws
ax.plot(xCopy[:,1],yHat)
plt.show()
'''

yHat = xMat * ws
print corrcoef(yHat.T, yMat)

# 8.2 局部加权线性回归 （LWLR，Locally Weighted Linear Regression）
# 欠拟合问题
# 引入只含对角元素的W矩阵， 使用高斯核函数， k

Myregression.lwlr(xArr[0], xArr, yArr, 1.0)
Myregression.lwlr(xArr[0], xArr, yArr, 0.001)
yHat = Myregression.lwlrTest(xArr, xArr, yArr, 1.0) # k=1.0, 0.01, 0.003

'''
xMat = mat(xArr)
srtInd = xMat[:,1].argsort(0)
xSort=xMat[srtInd][:,0,:]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:,1],yHat[srtInd])
ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0], s=2, c='red')

plt.show()
'''

# 8.3 示例：预测鲍鱼的年龄
'''
abX,abY = Myregression.loadDataSet('abalone.txt')
yHat01 = Myregression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
yHat1 = Myregression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
yHat10 = Myregression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)

print Myregression.rssError(abY[0:99],yHat01.T)
print Myregression.rssError(abY[0:99],yHat1.T)
print Myregression.rssError(abY[0:99],yHat10.T)

yHat01 = Myregression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
yHat1 = Myregression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
yHat10 = Myregression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)

print Myregression.rssError(abY[100:199],yHat01.T)
print Myregression.rssError(abY[100:199],yHat1.T)
print Myregression.rssError(abY[100:199],yHat10.T)

ws = Myregression.standRegres(abX[0:99], abY[0:99])
yHat = mat(abX[100:199])*ws
print Myregression.rssError(abY[100:199],yHat.T.A)
'''

# 8.4 缩减系数来“理解”数据
# 数据的特征（feature）n > 样本数 m

# 8.4.1 岭回归
# 拉格朗日乘子 lamda

abX,abY = Myregression.loadDataSet('abalone.txt')
ridgeWeights= Myregression.ridgeTest(abX,abY)
print shape(ridgeWeights)

'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)

plt.show()
'''

# 8.4.2 Lasso

# 8.4.3 前向逐步回归

xArr,yArr = Myregression.loadDataSet('abalone.txt')
Myregression.stageWise(xArr,yArr, 0.01, 200)
Myregression.stageWise(xArr,yArr, 0.001, 5000)

xMat = mat(xArr); yMat=mat(yArr).T
xMat = Myregression.regularize(xMat)
yMean = mean(yMat,0)
yMat = yMat - yMean     
weights = Myregression.standRegres(xMat,yMat.T)
print weights.T


# 8.5 权衡偏差与方差

# 8.6 示例：预测乐高玩具套装的价格
# 8.6.1 收集数据： 使用Google 购物的API

lgX = []; lgY = []
# print Myregression.setDataCollect(lgX,lgY)
print Myregression.setDataCollect()
