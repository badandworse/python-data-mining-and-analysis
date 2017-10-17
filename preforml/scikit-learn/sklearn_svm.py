#%%
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

'''
数据处理
'''
#%%
#iris.data 有150个样本点，每个样本点有4个特征
iris=datasets.load_iris()

#取出数据的两个纬度的特征
X=iris.data[:,[2,3]]
y=iris.target
#观察有哪些类别
np.unique(y)
#%%
#交叉检验，将样本中的30%拿出来检验样本
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
#%%
#数据标准化
#sklearn 中的 StandardScaler
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
#算出数据每一特征的样本平均值和标准差
sc.fit(X_train)  
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

'''
绘图函数
'''
#%%
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
    #setup marker generator and color map
    markers=('s','x','o','^','v')
    colors=('red','blue','yellow','grey','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])

    #plot the decision surface
    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))

    Z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    print(Z.shape)
    Z=Z.reshape(xx1.shape)
    print(xx1.shape)
    #等高线作图
    plt.contourf(xx1,xx2,Z,slpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    #plot all samples
    X_test,y_test=X[test_idx,:],y[test_idx]
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c='black',marker=markers[idx],label=cl)

    # highlight test samples
    if test_idx:
        X_test,y_test=X[test_idx,:],y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],c='',alpha=1.0,linewidths=1,marker='o',s=55,label='test set')
 



#%%
from sklearn.svm import SVC
svm=SVC(kernel='linear',C=1.0,random_state=0)
svm.fit(X_train_std,y_train)
y_pred=svm.predict(X_test_std)

#%%
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
plot_decision_regions(X=X_combined_std,y=y_combined,classifier=svm,test_idx=range(105,150))



'''
big data:
sklearn的SGDClassifier利用随机梯度
partial_fit可用于数据集很大时，部分匹配
'''

#初始化随机梯度下降版本的感知机、lr、svm   123456
from sklearn.linear_model import SGDClassifier
ppn=SGDClassifier(loss='perceptron')
lr=SGDClassifier(loss='log')
svm=SGDClassifier(loss="hinge")


'''
使用核svm解决非线性
'''

#创建数据集
np.random.seed(0)
X_xor=np.random.randn(200,2)
X_xor.shape
y_xor=np.logical_xor(X_xor[:,0]>0,X_xor[:,1]>0)
'''
np.where(condition,x,y)类似与三元运算符，如果条件为真，则返回想，
否则返回y
'''
y_xor=np.where(y_xor,1,-1)
y_xor.shape
#%%
sns.set()
plt.scatter(X_xor[y_xor==1,0],X_xor[y_xor==1,1],
            c='b',marker='x',label=1)
            
plt.scatter(X_xor[y_xor==-1,0],X_xor[y_xor==-1,1],
            c='b',marker='s',label=1)

svm=SVC(kernel='rbf',random_state=0,gamma=1.0,C=10)
svm.fit(X_xor,y_xor)
#%%
sns.set()
plot_decision_regions(X_xor,y_xor,classifier=svm)
plt.legend(loc='upper left')
plt.show()


'''
gamma参数的理解：
高斯球面的阶段参数，如果增大gamma值，会产生更加soft的决策界
即越大，对训练集分类效果越好，但是泛化能力越差
'''
svm=SVC(kernel='rbf',random_state=0,gamma=0.2,C=1.0)
svm.fit(X_test_std,y_test)
#%%
plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length[standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

#增大gamma值
#%%
svm=SVC(kernel='rbf',random_state=0,gamma=100,C=1.0)
svm.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length[standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()


