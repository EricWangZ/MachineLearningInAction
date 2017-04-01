
#coding=utf-8

#pip install feedparser

from sys import path
path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch04')
#path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch04')

# 第四章 基于概率论的分类方法： 朴素贝叶斯

# 4.1 基于贝叶斯决策理论的分类方法
# 如果 p1(x,y) > p2(x,y), 那么类别为  1 ；
# 如果p2（x,y) > p1(x,y), 那么类别为 2 ；

# 4.2 条件概率

# 4.3 使用条件概率来分类

# 4.4 使用朴素贝叶斯进行文档分类

# 4.5 使用Python进行文本分类

# 4.5.1 准备数据： 从文本中构建向量
import MyBayes
listOPosts,listClasses = MyBayes.loadDataSet()
myVocablist = MyBayes.createVocabList(listOPosts)
print myVocablist


print MyBayes.setOfWords2Vec(myVocablist, listOPosts[0])
print MyBayes.setOfWords2Vec(myVocablist, listOPosts[3])
#print MyBayes.bagOfWords2VecMN(myVocablist, listOPosts[0])
#print MyBayes.bagOfWords2VecMN(myVocablist, listOPosts[3])

trainMat = []
for postinDoc in listOPosts:
    trainMat.append(MyBayes.setOfWords2Vec(myVocablist,postinDoc))

''' 
    trainMat
    [[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1], 
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0], 
     [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], 
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1], 
     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]]
        
'''

print trainMat
print listClasses

# 4.5.2 训练算法： 从词向量计算概率
    
p0v,p1v,pAb = MyBayes.trainNB0(trainMat,listClasses)

print p0v
print p1v
print pAb

# 4.5.3 测试算法： 根据现实情况修改分类器

# reload(MyBayes)
MyBayes.testingNB()

# 4.5.4 准备数据： 文档词袋模型

''' test purpose
print MyBayes.textParse('this is my book.')  # test purpose
emailText = open('email/ham/6.txt').read()
listOfTokens = MyBayes.textParse(emailText)
print listOfTokens

'''

# 4.6 示例： 使用朴素贝叶斯过滤垃圾邮件
# 4.6.1 准备数据： 切分文本
# 4.6.2 测试算法： 使用朴素贝叶斯进行交叉验证

MyBayes.spamTest()


# 4.7 示例： 使用朴素贝叶斯分类器从个人广告中获取区域倾向

# 4.7.1 收集数据： 导入RSS源
# 4.7.2 分析数据： 显示地域相关的用词
import feedparser

ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')

vocabList,pSF,pNY=MyBayes.localWords(ny,sf)

print vocabList,pSF,pNY

print MyBayes.getTopWords(ny,sf)


