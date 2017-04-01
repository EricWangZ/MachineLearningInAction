
#coding=utf-8

from sys import path
path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch12')
#path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch12')

from numpy import *
import myFPGrowth

# 第十二章 使用FP-growth算法来高效发现频繁项集
# 1.构建FP树； 2.从FP树中挖掘频繁项集

# 12.1 FP树：用于编码数据集的有效方式
# FP： Frequent Pattern

# 12.2 构建FP树
# 12.2.1 创建FP树的数据结构
rootNode = myFPGrowth.treeNode('pyramid', 9, None)

rootNode.children['eye'] = myFPGrowth.treeNode('eye', 13, None)

rootNode.disp()

rootNode.children['phoenix'] = myFPGrowth.treeNode('phoenix', 3, None)

rootNode.disp()

# 12.2.2 构建FP树
# 第一次遍历去掉不满足最小支持度的元素项
# 第二次遍历构建FP树，从空集开始，如果树中已经存在现有元素，则增加现有元素的值；如果现有元素不存在，则添加一个分枝

simDat = myFPGrowth.loadSimpDat()
print simDat

initSet = myFPGrowth.createInitSet(simDat)
print initSet

myFPtree, myHeaderTab = myFPGrowth.createTree(initSet, 3)
myFPtree.disp()

# 12.3 从一棵FP树中挖掘频繁项集
# 12.3.1 抽取条件模式基
# 条件模式基conditional pattern base： 以所查找元素项为结尾的路径集合（前缀路径），再加上计数值

#print myFPGrowth.findPrefixPath('x', myHeaderTab['x'][1])
#print myFPGrowth.findPrefixPath('z', myHeaderTab['z'][1])
#print myFPGrowth.findPrefixPath('r', myHeaderTab['r'][1])
print myFPGrowth.findPrefixPath('t', myHeaderTab['t'][1])

# 12.3.2 创建条件FP树
# 
freqItems = []
myFPGrowth.mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
print freqItems

# 12.4 示例： 在twitter源中发现一些共现词

# 12.5 示例： 从新闻网站点击流中挖掘
parsedDat = [line.split() for line in open('kosarak.dat').readlines()]
initSet = myFPGrowth.createInitSet(parsedDat)
myFPtree, myHeaderTab = myFPGrowth.createTree(initSet, 100000)
myFreqList = []
myFPGrowth.mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)
print myFreqList




 