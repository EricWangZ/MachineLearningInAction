#pip install feedparser

#coding=utf-8

from sys import path
#path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch03')
path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch04')


import MyBayes
listOPosts,listClasses = MyBayes.loadDataSet()
myVocablist = MyBayes.createVocabList(listOPosts)
print myVocablist


print MyBayes.setOfWords2Vec(myVocablist, listOPosts[0])
print MyBayes.setOfWords2Vec(myVocablist, listOPosts[3])

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
    
p0v,p1v,pAb = MyBayes.trainNB0(trainMat,listClasses)

print p0v
print p1v
print pAb

'''
reload(MyBayes)
MyBayes.testingNB()

MyBayes.spamTest()



import feedparser
ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
ny=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')

ny['entries']
'''