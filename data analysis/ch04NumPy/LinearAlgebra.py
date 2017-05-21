import numpy as np


#numpy.linalg有很多函数
x=np.array([[1.,2.,3.],[4.,5.,6.]])
y=np.array([[6.,23.],[4.,3.],[6.,9.]])
x.dot(y) #2*3 矩阵*  3*2维 得到2*2 维

np.dot(x,np.ones(3))
