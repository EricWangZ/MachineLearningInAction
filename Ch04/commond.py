#pip install feedparser

#coding=utf-8

from sys import path
path.append(r'C:\Users\eyuiwng\Desktop\Study\machine learning\workspace\Ch04')
#path.append(r'D:\Study\Workspaces\MyEclipse 2015\MachineLearningInAction\Ch04')


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
    
p0v,p1v,pAb = MyBayes.trainNB0(trainMat,listClasses)

print p0v
print p1v
print pAb

# reload(MyBayes)
MyBayes.testingNB()

''' test purpose
print MyBayes.textParse('this is my book.')  # test purpose
emailText = open('email/ham/6.txt').read()
listOfTokens = MyBayes.textParse(emailText)
print listOfTokens

'''

MyBayes.spamTest()


import feedparser

ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')

vocabList,pSF,pNY=MyBayes.localWords(ny,sf)

print vocabList,pSF,pNY

print MyBayes.getTopWords(ny,sf)


