
#coding=utf-8

from sys import path
path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch11')
#path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch11')

from numpy import *
import myApriori

# 第十一章 使用Anroiri算法进行关联分析
# 关联分析或关联规则学习

# 11.1 关联分析
# 关联分析是一种在大规模数据集中寻找有趣关系的任务： 频繁项集 （frequent item set） 和 关联规则 （association rules）
# 支持度 （support）， 最小支持度。 {啤酒，尿布}支持度 3/5, 支持度 {尿布} 4/5
# 可信度 （confidence） , "支持度{啤酒，尿布} / 支持度 {尿布}" --> “尿布 --> 啤酒”的可信度 3/4


# 11.2 Apriori 原理
# 如果某个项集是频繁的，那么它的所有子集也是频繁的。


# 11.3 使用Apriori算法来发现频繁集
# 满足（大于）某最小支持度的集合，就是频繁集
# 11.3.1 生成候选项集
dataSet = myApriori.loadDataSet()
print dataSet

C1 = myApriori.createC1(dataSet)
D = map(set,dataSet)

L1, suppData0 = myApriori.scanD(D,C1,0.5)

print L1, suppData0

# 11.3.2 组织完整的Apriori算法

L, suppData = myApriori.apriori(dataSet)
print L, suppData

print myApriori.aprioriGen(L[0],2)
L, suppData = myApriori.apriori(dataSet, minSupport=0.7)
print L, suppData

# 11.4 从频繁项集中挖掘关联规则
# 对于关联规则，我们采用量化指标：可信度 support (P|H) / Support (P)

L, suppData = myApriori.apriori(dataSet, minSupport=0.5)
rules = myApriori.generateRules(L,suppData, minConf=0.7)
print rules

rules = myApriori.generateRules(L,suppData, minConf=0.5)
print rules

# 11.5 示例： 发现国会投票中的模式

# 11.6 示例： 发现毒蘑菇的相似特征

mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
L, suppData = myApriori.apriori(mushDatSet, minSupport=0.3)
for item in L[1]:
    if item.intersection('2'): print item
    
for item in L[3]:
    if item.intersection('2'): print item

