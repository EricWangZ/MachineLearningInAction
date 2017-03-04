#coding=utf-8

#pip install feedparser

#from sys import path
#path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch04')

# 4.1 基于贝叶斯决策理论的分类方法
# 4.2 条件概率
# 4.3 使用条件概率来分类
# 4.4 使用朴素贝叶斯进行文档分类
# 4.5 使用python进行文本分类
# 4.5.1 准备数据：从文本中构建词向量

import bayes
listOPosts,listClasses = bayes.loadDataSet()
myVocablist = bayes.createVocabList(listOPosts)
print myVocablist

bayes.setOfWords2Vec(myVocablist, listOPosts[0])
bayes.setOfWords2Vec(myVocablist, listOPosts[3])

# 4.5.2 训练算法：从词向量计算概率
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocablist,postinDoc))

print trainMat
print listClasses
    
p0v,p1v,pAb = bayes.trainNB0(trainMat,listClasses)

print p0v
print p1v
print pAb

# 4.5.3 测试算法：根据现实情况修改分类器
reload(bayes)
bayes.testingNB()
# 4.5.4 准备数据：文档词袋模型

# 4.6 示例：使用朴素贝叶斯过滤垃圾邮件

# 4.6.1 准备数据：切分文本
# 4.6.2 测试算法： 使用朴素贝叶斯继续交叉验证
bayes.spamTest()

# 4.7示例 使用朴素贝叶斯分类器从个人广告中获取区域倾向

# 4.7.1 收集数据： 导入RSS源

import feedparser
ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
ny=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')

ny['entries']

# 4.7.2 分析数据： 显示地域相关的用词