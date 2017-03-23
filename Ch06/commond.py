#coding=utf-8

from sys import path
path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch06')
#path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch06')

from numpy import *
import MysvmMLiA

# 第六章 支持向量机 （SVM，Support Vector Machines）
# 部分认为， SVM是最好的现成的分类器。 SVM有很多实现，这里使用“序列最小优化” （SMO， Sequential Minimal Optimization)

# 6.1 基于最大间隔分隔数据
# 6.2 寻找最大间隔
# 6.2.1 分类器求解的优化问题
# 6.2.2 SVM应用的一般框架

# 6.3 SMO 高效优化算法
# 6.3.1 Platt的SMO算法

# 6.3.2 应用简化版SMO算法处理小规模数据集

dataArr,labelArr = MysvmMLiA.loadDataSet('testSet.txt')
labelArr

b,alphas = MysvmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)

print b
print alphas[alphas>0]
print alphas
print shape(alphas[alphas>0])

for i in range(100):
    if alphas[i]>0.0: print dataArr[i],labelArr[i]

# 6.4 利用完整platt SMO 算法加速优化
dataArr,labelArr = MysvmMLiA.loadDataSet('testSet.txt')
b,alphas = MysvmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
print b, alphas
print alphas[alphas>0]

# 计算超平面
ws = MysvmMLiA.calcWs(alphas, dataArr, labelArr)
print ws

# 使用w进行分类
datMat = mat(dataArr)
datMat[0] * mat(ws) + b 
print labelArr[0]

datMat[2] * mat(ws)+ b 
print labelArr[2]

datMat[1] * mat(ws)+ b 
print labelArr[1]

# 6.5 在复杂数据上使用核函数
# 6.5.1 利用核函数将数据映射到高维空间，经过空间转换之后，我们可以在高维空间中解决线性问题，等价于在低维空间中解决非线性问题
# 6.5.2 径向基核函数 （高斯核函数）

# MysvmMLiA.testRbf()

# 6.6 示例： 手写识别问题回顾

#MysvmMLiA.testDigits(('rbf', 10))

MysvmMLiA.testDigits(('rbf', 20))

