#coding=utf-8

from sys import path
#path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch03')
path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch03')


import myTrees
myDat, labels = myTrees.createDataSet()
print myDat,labels

'''
myDat (the result is in the last column)
    [[1, 1, 'yes'], 
     [1, 1, 'yes'], 
     [1, 0, 'no'], 
     [0, 1, 'no'], 
     [0, 1, 'no']] 
labels (the name of features)  
    ['no surfacing', 'flippers']
'''

print myTrees.calcShannonEnt(myDat)

'''
myDat [0][-1] = 'no'
myDat [1][-1] = 'no'
myDat [2][-1] = 'yes'
print myDat
print myTrees.calcShannonEnt(myDat)
'''


print myTrees.splitDataSet(myDat,1,1)
# [[1, 'yes'], [1, 'yes'], [0, 'no'], [0, 'no']], took out column 2
print myTrees.splitDataSet(myDat,0,0)
# [[1, 'no'], [1, 'no']], took out column 1

print myTrees.chooseBestFeatureToSplit(myDat)
# 0, the first feature is the best choice for split

'''

myTree = trees.createTree(myDat,labels)
print myTree


# 3.2 ä½¿ç”¨Matplotlibæ³¨è§£ç»˜åˆ¶æ ‘å½¢å›¾

# 3.2.1 Matplotlib æ³¨è§£

import treePlotter
treePlotter.createPlot(myTree)

'''
