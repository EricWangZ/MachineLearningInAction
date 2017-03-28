
#coding=utf-8

from sys import path
path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch10')
#path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch10')

from numpy import *
import myKMeans

# 第三部分 无监督学习
# 第十章 利用K-均值聚类算法对未标注数据分组
# 聚类 Clustering
# Cluster Identification

# 10.1 K-均值聚类算法
# 随机找几个质心，把所有数据分类到这些质心上（根据距离大小），称之为簇。计算簇的平均值为新的质心。循环下去

datMat = mat(myKMeans.loadDataSet('testSet.txt'))
'''
print datMat

print min(datMat[:,0])]
print max(datMat[:,0])
print min(datMat[:,1])
print max(datMat[:,1])

print myKMeans.randCent(datMat, 2)

print myKMeans.distEclud(datMat[0],datMat[2])
'''
myCentroids, clustAssign = myKMeans.kMeans(datMat, 4)
# print clustAssign

# 10.2 使用后处理来提高聚类性能
# SSE（Sum of Squared Error), 误差平方和
# 将最大SSE的簇的点过滤出来，再在这些点上运行K-均值（k=2）分成两个簇
# 同时合并两个簇，以保证总簇数不变。1） 合并质心距离最近的两个簇； 2）合并两个使得SSE增幅最小的质心

# 10.3 二分 K-均值算法
datMat = mat(myKMeans.loadDataSet('testSet2.txt'))
myCentroids, clustAssign = myKMeans.biKMeans(datMat, 3)

# 10.4 示例： 对地图上的点进行聚类
# 10.1 Yahoo! PlaceFounder API

# geoResults = myKMeans.geoGrab('1 VA Center', 'Augusta, ME')

# print myKMeans.massPlaceFind('portlandClubs.txt')


# 10.2 对地理坐标进行聚类

myKMeans.clusterClubs(5)















