
#coding=utf-8

from sys import path
#path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch07')
path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch07')

from numpy import *
import Myadaboost

# 第七章 利用AdaBoost元算法提高分类性能
# 7.1 基于数据集多重抽样的分类器
# 7.1.1 bagging： 基于数据随机重抽样的分类器构建方法
# 7.1.2 boosting： 


# 7.2 训练算法： 基于错误提升分类器的性能

# 7.3 基于单层决策树构建弱分离器

dataMat,classLabels = Myadaboost.loadSimpData()

D = mat(ones((5,1))/5)
bestStump,minError,bestClasEst = Myadaboost.buildStump(dataMat,classLabels,D)
print bestStump,minError,bestClasEst


# 7.4 完整Adaboost算法的实现

weakClassArr,aggClassEst = Myadaboost.adaBoostTrainDS(dataMat,classLabels,9)
print weakClassArr
print aggClassEst


dataMat,classLabels = Myadaboost.loadSimpData()
classifierArray, aggClassEst = Myadaboost.adaBoostTrainDS(dataMat,classLabels,30)

# 7.5 测试算法：基于AdaBoost的分类
Myadaboost.adaClassify ([0,0], classifierArray)
Myadaboost.adaClassify ([[5,5],[0,0]], classifierArray)

# 7.6 示例：在一个难数据集上应用AdaBoost
dataArr,labelArr = Myadaboost.loadDataSet("horseColicTraining2.txt")
classifierArray,aggClassEst = Myadaboost.adaBoostTrainDS(dataArr,labelArr)
print classifierArray

testArr,testLabelArr = Myadaboost.loadDataSet("horseColicTest2.txt")
prediction10=Myadaboost.adaClassify (testArr, classifierArray)

errArr = mat(ones((67,1)))
print errArr[prediction10 != mat(testLabelArr).T].sum()/67




