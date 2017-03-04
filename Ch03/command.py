#coding=utf-8

#from sys import path
#path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch03')

#Ch3 决策树

# 3.1 决策树的构造

# 3.1.1 信息增益

# 计算给定数据集的香农熵

import trees
myDat, labels = trees.createDataSet()
print myDat

print trees.calcShannonEnt(myDat)

'''
myDat [0][-1] = 'no'
myDat [1][-1] = 'no'
myDat [2][-1] = 'yes'
print myDat
print trees.calcShannonEnt(myDat)
'''


# 3.1.2 划分数据集

print trees.splitDataSet(myDat,0,1)
print trees.splitDataSet(myDat,0,0)

# 选择最好的数据集划分方式
print trees.chooseBestFeatureToSplit(myDat)

# 3.1.3 递归构建决策树

myTree = trees.createTree(myDat,labels)
print myTree


# 3.2 使用Matplotlib注解绘制树形图

# 3.2.1 Matplotlib 注解

import treePlotter
treePlotter.createPlot(myTree)

