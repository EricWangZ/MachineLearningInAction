#coding=utf-8

#from sys import path
#path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch03')


import myTrees
myDat, labels = myTrees.createDataSet()
print myDat

print myTrees.calcShannonEnt(myDat)

'''
myDat [0][-1] = 'no'
myDat [1][-1] = 'no'
myDat [2][-1] = 'yes'
print myDat
print trees.calcShannonEnt(myDat)
'''


print myTrees.splitDataSet(myDat,1,1)
# print trees.splitDataSet(myDat,0,0)

print myTrees.chooseBestFeatureToSplit(myDat)

'''
# 3.1.3 é€’å½’æž„å»ºå†³ç­–æ ‘

myTree = trees.createTree(myDat,labels)
print myTree


# 3.2 ä½¿ç”¨Matplotlibæ³¨è§£ç»˜åˆ¶æ ‘å½¢å›¾

# 3.2.1 Matplotlib æ³¨è§£

import treePlotter
treePlotter.createPlot(myTree)

'''
