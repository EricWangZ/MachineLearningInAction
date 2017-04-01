#coding=utf-8

from sys import path
#path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch03')
path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch03')


import myTrees

# 第三章 决策树
# 3.1 决策树的构造

# 3.1.1 信息增益
# 熵

'''
myDat # (the result is in the last column)
    [[1, 1, 'yes'], 
     [1, 1, 'yes'], 
     [1, 0, 'no'], 
     [0, 1, 'no'], 
     [0, 1, 'no']] 
labels (the name of features)  
    ['no surfacing', 'flippers']
'''
myDat, labels = myTrees.createDataSet()
print myDat,labels
print myTrees.calcShannonEnt(myDat)

'''
myDat [0][-1] = 'no'
myDat [1][-1] = 'no'
myDat [2][-1] = 'yes'
print myDat
print myTrees.calcShannonEnt(myDat)
'''

# 3.1.2 划分数据集
print myTrees.splitDataSet(myDat,1,1)
# [[1, 'yes'], [1, 'yes'], [0, 'no'], [0, 'no']], took out column 2
print myTrees.splitDataSet(myDat,0,0)
# [[1, 'no'], [1, 'no']], took out column 1

print myTrees.chooseBestFeatureToSplit(myDat)
# 0, the first feature is the best choice for split

# 3.1.3 递归构建决策树
subLabels = labels[:]  # create a new label, if use subLabels = labels, then reference to the same data
myTree = myTrees.createTree(myDat,subLabels)
print myTree
print labels
#{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
'''
        print myTree['no surfacing']
        print myTree['no surfacing'][0]
        print myTree['no surfacing'][1]
        yourtree = {3: {4:{5:{6}}}}
        print yourtree
'''

# 3.2 在Python中使用Matplotlib注解绘制树形图
import treePlotter
treePlotter.createPlot(myTree)

myTree['no surfacing'][2]='maybe'
print myTree
treePlotter.createPlot(myTree)

# 3.3 测试和存储分类器
# 3.3.1 测试算法： 使用决策树执行分类
myTree = treePlotter.retrieveTree(0)
print myTree
print myTrees.classify(myTree,labels,[1,0])
print myTrees.classify(myTree,labels,[1,1])

# 3.3.2 使用算法： 决策树的存储
myTrees.storeTree(myTree,'ClassifierStorage.txt')
yourTree = myTrees.grabTree('ClassifierStorage.txt')
print yourTree

# 3.4 示例： 使用决策树预测隐形眼睛类型
fr=open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = myTrees.createTree(lenses, lensesLabels)
print lensesTree
treePlotter.createPlot(lensesTree)




