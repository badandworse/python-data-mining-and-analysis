#%%
import numpy as np
from numpy import loadtxt,where,transpose
from numpy import zeros,ones
from pylab import scatter,show,legend,xlabel,ylabel,plot
import math
from math import log
#load the dataset
#%%
# 加载数据，数据的分隔符是','
data=loadtxt('data1.txt',delimiter=',')
X=data[:,0:2]
# 切片第二个数据如果是一个，
# 那将是把特定列的一个数据取出来，形成一个单独的列
# 因此需要切片
# 切片0:2 会取出0，1列数据，而不包括2
data[:,0:2]
y=data[:,2]

#%%
# 选出所有y=1的值的索引
pos=where(y==1)
neg=where(y==0)
scatter(X[pos,0],X[pos,1],marker='o',c='b')
scatter(X[neg,0],X[neg,1],marker='x',c='r')
xlabel('Feature1/Exam 1 score')
ylabel('Feature2/Exam 2 score')
legend(['Fail','Pass'])
show()
# shape返回一个元祖，
# 表示元素纬度，
# 每个值代表该维度上有多少个值
X.shape[0]

#逻辑斯蒂回归(logistic regression,re)
##sigmoid函数，代价函数，梯度下降

#%%
#sigmod函数
def sigmod(X):
    den=1.0+ np.exp(-1.0*X)
    gz=1.0/den
    return gz

#%%
#代价函数
def costFunction(theta,X,y):
    m=y.size
    h=sigmod(X.dot(theta))
    J=-1.0*(1./m)*(y.T.dot(np.log(output))+(1-y).T.dot(np.log(1-output)))
    if np.isnan(J):
        return(np.inf)
    return J

#%%
#计算梯度
def gradient(X,y,opts):
    #第一个是样本数，第二个是样本特征数
    numSamples,numFeatures=shape(X)
    alpha=opts['alpha']
    maxIter=opts['maxIter']
    weights=ones(numFeatures)
    now_cost=costFunction(weights,X,y)
    i=100
    print(1)
    for i in range(maxIter):
        output=sigmod(X.dot(weights))
        error=output-y
        weights=weights-alpha*X.T.dot(error.T)
        print(weights)

    '''
    while i>opts['minUp']:  
        output=sigmod(X.dot(weights))
        error=output-y
        weights=weights-alpha*X.T.dot(error.T)
        next_cost=costFunction(weights,X,y)
        i=abs(next_cost-cost)
        now_cost=next_cost
    '''
    return weights


#%%
def predict(theta,X):
    '''Predict label using learned logistic regression parameters'''
    m,n=X.shape
    p=zeros(X.shape[0])
    h=sigmod(X.dot(theta))
    for it in range(0,h.shape[0]):
        if h[it]>0:
            p[it]=1
        else:
            p[it]=0
    return p




#%%
X.dot(initial_theta)
zeros((4,1))
weights=zeros(X.shape[1])
weights
X.dot(weights) 
X.dot(weights).size
output=sigmod(X.dot(weights))
error=output-y
error
X.T
X.T.dot(y.T)
#%%
opts=dict()
opts['alpha']=0.01
opts['maxIter']=5000
opts['minUp']=0.000000005
weights=gradient(X,y,opts)
print(1)

#[ 35.51156489 -12.49598229]
#%%
weights
p=predict(weights,X)
y[where(p==y)].size/float(y.size)
g=p-y
g[g==0].size



#%%
# 选出所有y=1的值的索引
pos=where(y==1)
neg=where(y==0)
scatter(X[pos,0],X[pos,1],marker='o',c='b')
scatter(X[neg,0],X[neg,1],marker='x',c='r')
xlabel('Feature1/Exam 1 score')
ylabel('Feature2/Exam 2 score')
legend(['Fail','Pass'])
show()
weights

