#coding=utf-8


#from sys import path
#path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch05')

import os
#os.getcwd()
os.chdir('C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch05')

#第五章 Logistic回归

# 5.1基于Logistic回归和Sigmoid函数的分类

# 5.2 基于最优化方法的最佳回归系数确定
#     z=w0x0+w1x1+w2x2+...+wnxn

#5.2.1 梯度上升法
#     求导数，w：=w+a*d(f(w))，迭代一定次数来求得最优值，类似最优化算法；

#5.2.2 训练算法：使用梯度上升找到最佳参数
# 在整个数据集上迭代500次

from numpy import *
import logRegres

dataArr, labelMat=logRegres.loadDataSet()
weights=logRegres.gradAscent(dataArr, labelMat)

#5.2.3 分析数据：画出决策边界
logRegres.plotBestFit(weights.getA())

#5.2.4 训练算法：随机梯度上升
# 在整个数据集上遍历一次
weights=logRegres.stocGradAscent0(array(dataArr), labelMat)
logRegres.plotBestFit(weights)

# 改进的随机梯度算法
weights=logRegres.stocGradAscent1(array(dataArr), labelMat)
logRegres.plotBestFit(weights)


# 5.3 从疝气病预测病马的死亡率

# 5.3.1 准备数据： 处理数据中的缺失值
#   1） 用0替代特征值的缺失值； 2） 如果类别标签缺失，将该条数据丢弃

# 5.3.2 测试算法： 用logistic回归进行分类

reload(logRegres)
logRegres.multiTest()



