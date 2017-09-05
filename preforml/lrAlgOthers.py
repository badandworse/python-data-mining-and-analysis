#%%
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from sklearn.preprocessing import PolynomialFeatures


#%%
#读取数据的函数
def loaddata(file,delimiter):
    data=np.loadtxt(file,delimiter=delimiter)
    print('Dimensions:',data.shape)
    print(data[1:6,:])
    return (data)

#%%
#画图
def plotData(data,label_x,label_y,label_pos,label_neg,axes=None):
    #获得正负样本的下标
    neg=data[:,2]==0
    pos=data[:,2]==1

    if axes==None:
        axes=plt.gca()
    
    axes.scatter(data[pos][:,0],data[pos][:,1],marker='+',c='k',s=60,linewidth=2,label=label_pos)
    axes.scatter(data[neg][:,0],data[neg][:,1],c='y',s=60,label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True,fancybox=True)

#%%
data=loaddata('data1.txt',',')
data
#加入一项为0是加入稀疏项，bias项
X=np.c_[np.ones((data.shape[0],1)),data[:,0:2]]
y=np.c_[data[:,2]]
y
plotData(data,'Exam 1 score','Exam 2 score','Pass','Fail')

#逻辑斯蒂回归
#定义sigmoid函数
#%%
def sigmoid(x):
    return 1/(1+np.exp(-x))

#定义损失函数
#%%
def costFunction(theta,X,y):
    m=y.size
    h=sigmoid(X.dot(theta))

    J=-1.0*(1.0/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))
    if np.isnan(J[0]):
        return(np.inf)
    return J[0]

#求解梯度
#%%
def gradient(theta,X,y):
    m=y.size
    h=sigmoid(X.dot(theta.reshape(-1,1)))
    grad=(1.0/m)*X.T.dot(h-y)
    return (grad.flatten())


#%%
initial_theta=np.zeros(X.shape[1])
cost=costFunction(initial_theta,X,y)
grad=gradient(initial_theta,X,y)
print('Cost:\n',cost)
print('Grad:\n',grad)
print(initial_theta)

#用minimize求最小化值
# 输入决定函数，自变量，
# args是函数需要用到的参数,
# jac传入一个梯度的计算方法。
#  If jac is a Boolean and is True, fun is assumed to return the gradient along with the objective function. 
#  If False, the gradient will be estimated numerically
#  it must accept the same arguments as fun
res=minimize(costFunction,initial_theta,args=(X,y),jac=gradient,options={'maxiter':400})
res

#%%
def predict(theta,X,threshold=0.5):
    p=sigmoid(X.dot(theta.T))>=threshold
    return p.astype('int')

p=predict(res.x,X)
p.shape
y.shape
y=y.flatten()
y[where(p==y)].size


#画出边界
#%% 
plt.scatter(45,85,s=60,c='r',marker='v',label='(45,85)')
plotData(data,'Exam 1 score','Exam 2 score','Admitted','Not admitted')
x1_min,x1_max=X[:,1].min(),X[:,1].max()
x2_min,x2_max=X[:,2].min(),X[:,2].max()
x1_min,x1_max
xx1,xx2=np.meshgrid(np.linspace(x1_min,x1_max),np.linspace(x2_min,x2_max))
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)

# 画出指定等高线值的等高线，
# 下面的例子是画出值为0.2，0.5的等高线
plt.contour(xx1, xx2, h, [0.2,0.5], linewidths=1, colors='b')
plt.legend(loc='upper right')


#%%
def f(x,y):
    return (1-x/2+x**5+y**5)*np.exp(-x**2-y**2)

#%%
n=256
x=np.linspace(-3,3,n)
y=np.linspace(-3,3,n)
X,Y=np.meshgrid(x,y)

# 用counter
plt.contour(X,Y,f(X,Y),[0.5],8)

#加正则化项的逻辑斯特回归
data2=loaddata('data2.txt',',')
y=np.c_[data2[:,2]]
x=data2[:,0:2]

plotData(data2,'Microchip Test 1','Microchip Test 2','y=1','y=0')
poly=PolynomialFeatures(6)
XX=poly.fit_transform(data2[:,0:2])
XX
data2[:,0:2].shape
XX.shape

#%%
# 损失函数
def costFunctionReg(theta,reg,*args):
    m=y.size
    h=sigmod(XX.dot(theta))
    J=-1.0*(1.0/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))+(reg/(2.0*m))*np.sum(np.square(theta[1:]))
    if np.isnan(J[0]):
        return(np.inf)
    return J[0]

#%%
# 定义损失函数
def costFunctionReg(theta, reg, *args):
    m = y.size
    h = sigmoid(XX.dot(theta))
    
    J = -1.0*(1.0/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + (reg/(2.0*m))*np.sum(np.square(theta[1:]))
    
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])
#%%
# 梯度计算
def gradientReg(theta,reg,*args):
    m=y.size
    h=sigmoid(XX.dot(theta.reshape(-1,1)))
    grad=(1.0/m)*XX.T.dot(h-y)+(reg/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]
    return grad.flatten()

#%%
def gradientReg(theta, reg, *args):
    m = y.size
    h = sigmoid(XX.dot(theta.reshape(-1,1)))
      
    grad = (1.0/m)*XX.T.dot(h-y) + (reg/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]
        
    return(grad.flatten())


#%%
# reshape 中-1表示我懒得计算这个值是多少
# 请根据后面的值和需要重塑的数组来得到这个值
initial_theta2=np.zeros(XX.shape[1])
initial_theta2

costFunctionReg(initial_theta2,1,XX,y)

# 共享y轴
fig,axes=plt.subplots(1,3,sharey=True,figsize=(17,5))

# 决策边界，咱们分别来看看正则化系数lambda太大太小分别会出现什么情况
# Lambda = 0 : 就是没有正则化，这样的话，就过拟合咯
# Lambda = 1 : 这才是正确的打开方式
# Lambda = 100 : 卧槽，正则化项太激进，导致基本就没拟合出决策边界
#%%
#C是正则化项的参数
for i, C in enumerate([0.0, 1.0, 100.0]):
    # 最优化 costFunctionReg
    res2 = minimize(costFunctionReg, initial_theta2, args=(C, XX, y), jac=gradientReg, options={'maxiter':3000})
    
    # 准确率
    accuracy = 100.0*sum(predict(res2.x, XX) == y.ravel())/y.size    

    # 对X,y的散列绘图
    plotData(data2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes.flatten()[i])
    
    # 画出决策边界
    x1_min, x1_max = X[:,0].min(), X[:,0].max(),
    x2_min, x2_max = X[:,1].min(), X[:,1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(res2.x))
    h = h.reshape(xx1.shape)
    axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');       
    axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), C))


np.sum(np.square(initial_theta2))