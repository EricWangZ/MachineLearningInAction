#coding=utf-8

from sys import path
#path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch05')
path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch05')

'''
import os
#os.getcwd()
os.chdir('C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch05')
'''

from numpy import *
import MylogRegres

dataArr, labelMat=MylogRegres.loadDataSet()
weights=MylogRegres.gradAscent(dataArr, labelMat)

print weights

#5.2.3 分析数据：画出决策边界
MylogRegres.plotBestFit(weights.getA())

#5.2.4 训练算法：随机梯度上升
# 在整个数据集上遍历一次
weights=MylogRegres.stocGradAscent0(array(dataArr), labelMat)
MylogRegres.plotBestFit(weights)

# 改进的随机梯度算法
weights=MylogRegres.stocGradAscent1(array(dataArr), labelMat)
MylogRegres.plotBestFit(weights)

# 5.3 从疝气病预测病马的死亡率

# 5.3.1 准备数据： 处理数据中的缺失值
#   1） 用0替代特征值的缺失值； 2） 如果类别标签缺失，将该条数据丢弃

# 5.3.2 测试算法： 用logistic回归进行分类

reload(MylogRegres)
MylogRegres.multiTest()


