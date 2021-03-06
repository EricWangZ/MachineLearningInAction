
#coding=utf-8


from sys import path
# path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\MachineLearningInAction\Ch02')
path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch02')

# 第二章 k-近邻算法

# 2.1 k-近邻算法概述
# 选择k个最相似数据中出现次数最多的分类，作为新数据的分类

# 2.1.1 准备：使用python导入数据

# 2.1.2 实施kNN算法

import myKNN
dataSet,labels = myKNN.createDataSet()
myKNN.classify0([0,0],dataSet,labels,3)

# 2.1.3 如何测试分类器

# 2.2 示例：使用k-近邻算法改进约会网站的配对效果
# 2.2.1 准备数据：从文本文件中解析数据
datingDataMat, datingLabels = myKNN.file2matrix('datingTestSet2.txt')

# 2.2.2 分析数据： 使用Matplotlib创建散点图
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
plt.show()


ax.scatter(datingDataMat[:,1], datingDataMat[:,0], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()

# 2.2.3 准备数据： 归一化数值
normMat, ranges, minVals = myKNN.autoNorm(datingDataMat)
# print normMat, ranges,minVals

# 2.2.4 测试算法： 作为完整程序验证分类器
reload(myKNN)
myKNN.datingClassTest()

# 2.2.5 使用算法；构建完整可用系统

myKNN.classifyPerson()

# 2.3 示例： 手写识别系统
# 2.3.1 准备数据： 将图像转换为测试向量

myKNN.img2vector('testDigits/0_13.txt')

# 2.3.2 测试算法： 使用k-近邻算法识别手写数字
myKNN.handwritingClassTest()


