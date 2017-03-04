#coding=utf-8

#from sys import path
#path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch03')

#Ch3 å†³ç­–æ ‘

# 3.1 å†³ç­–æ ‘çš„æž„é€ 

# 3.1.1 ä¿¡æ�¯å¢žç›Š

# è®¡ç®—ç»™å®šæ•°æ�®é›†çš„é¦™å†œç†µ

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


# 3.1.2 åˆ’åˆ†æ•°æ�®é›†

print trees.splitDataSet(myDat,0,1)
print trees.splitDataSet(myDat,0,0)

# é€‰æ‹©æœ€å¥½çš„æ•°æ�®é›†åˆ’åˆ†æ–¹å¼�
print trees.chooseBestFeatureToSplit(myDat)

# 3.1.3 é€’å½’æž„å»ºå†³ç­–æ ‘

myTree = trees.createTree(myDat,labels)
print myTree


# 3.2 ä½¿ç”¨Matplotlibæ³¨è§£ç»˜åˆ¶æ ‘å½¢å›¾

# 3.2.1 Matplotlib æ³¨è§£

import treePlotter
treePlotter.createPlot(myTree)

