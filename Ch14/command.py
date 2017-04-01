
#coding=utf-8

from sys import path
from numpy.dual import svd
path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch14')
#path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch14')

from numpy import *
import mySVDRec

# 第十四章 使用SVD来简化数据

# 14.1 SVD的应用
# 隐性语义索引应用于搜索和信息检索领域的，
# SVD 在推荐系统中的应用

# 14.1.1 隐性语义索引 Latent Semantic Indexing， LSI
# 14.1.2 推荐系统

# 14.2 矩阵分解
# SVD:   Data (mxn) = U(mxm) E(mxn) V.T(nxn)
# E 是对角阵，且对角元素从大到小排列，称之为奇异值singular value， 且r个以后后面大部分为0
# 奇异值 是 Data*Data.T的特征值的平方根

# 14.3 利用Python实现SVD

from numpy import *
U, Sigma, VT = linalg.svd([[1,1],[7,7]])
print U 
print Sigma 
print VT

Data = mySVDRec.loadExData()
U, Sigma, VT = linalg.svd(Data)
print Sigma

Sig3 = mat(([Sigma[0],0,0],[0,Sigma[1],0],[0,0,Sigma[2]]))
print Sig3
print U[:,:3]*Sig3*VT[:3,:]

# 14.4 基于协同过滤的推荐引擎 collaborative filtering

# 14.4.1 相似度的计算
# 使用用户对物品的意见来计算相似度；  
# 1. 相似度 = 1 / (1 + 欧氏距离）
# 2. 皮尔逊相关系数： Pearson correlation，  0.5 + 0.5 *corrcoef()
# 3. 余弦相似度 cosine similarity：  COS Theta = A * B / (|A|*|B|)

myMat = mat(mySVDRec.loadExData())
print mySVDRec.ecludSim(myMat[:,0],myMat[:,4])
print mySVDRec.ecludSim(myMat[:,0],myMat[:,0])

print mySVDRec.cosSim(myMat[:,0],myMat[:,4])
print mySVDRec.cosSim(myMat[:,0],myMat[:,0])

print mySVDRec.pearsSim(myMat[:,0],myMat[:,4])
print mySVDRec.pearsSim(myMat[:,0],myMat[:,0])

# 14.4.2 基于物品的相似度还是基于用户的相似度？

# 14.4.3 推荐引擎的评价

# 14.5 示例： 餐馆菜肴推荐引擎
# 首先构建一个基本的推荐引擎，寻找用户没有尝过的菜肴。然后，通过SVD来减少特征空间并提高推荐的效果
# 这之后，将程序打包并通过用户可读的人机界面提供给人们使用。最后，我们介绍在构建推荐系统时面临的一些问题

# 14.5.1 推荐未尝过的菜肴

myMat = mat(mySVDRec.loadExData())
myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
myMat[3,3]=2
print myMat

print mySVDRec.recommend(myMat,2)
print mySVDRec.recommend(myMat,2, simMeas=mySVDRec.ecludSim)
print mySVDRec.recommend(myMat,2, simMeas=mySVDRec.pearsSim)

# 14.5.2 利用SVD提高推荐效果
print 'SVD enhancement'
Data = mySVDRec.loadExData2()
U, Sigma, VT = linalg.svd(Data)
print Sigma

Sig2 = Sigma**2
sum(Sig2)*0.9
sum(Sig2[:2])
print sum(Sig2[:3])

print mySVDRec.recommend(mat(Data),1, estMethod=mySVDRec.svdEst)
print mySVDRec.recommend(mat(Data),1, estMethod=mySVDRec.svdEst, simMeas=mySVDRec.pearsSim)

# 14.5.3 构建推荐引擎面临的挑战

# 14.6 基于SVD的图像压缩
mySVDRec.imgCompress(2)



