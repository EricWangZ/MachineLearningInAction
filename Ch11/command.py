
#coding=utf-8

from sys import path
path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch11')
#path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch11')

from numpy import *
import myKMeans

# 第十一章 使用Anroiri算法进行关联分析
# 关联分析或关联规则学习

# 11.1 关联分析
# 关联分析是一种在大规模数据集中寻找有趣关系的任务： 频繁项集 （frequent item set） 和 关联规则 （association rules）
# 支持度 （support）， 最小支持度。 {啤酒，尿布}支持度 3/5, 支持度 {尿布} 4/5
# 可信度 （confidence） , "支持度{啤酒，尿布} / 支持度 {尿布}" --> “尿布 --> 啤酒”的可信度 3/4


# 11.2 Apriori 原理
# 如果某个项集是频繁的，那么它的所有子集也是频繁的。



