import kNN
group,labels = kNN.createDataSet()
group
'reload(kNN)'

datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
datingDataMat

import matplotlib
import matplotlib.pyplot as plt
from numpy import *
fig = plt.figure()
ax = fig.add_subplot(111)
'ax.scatter(datingDataMat[:,1], datingDataMat[:,2])'
'plt.show()'

ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()

            
reload(kNN)
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
normMat
ranges
minVals
