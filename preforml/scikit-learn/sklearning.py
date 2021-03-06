#%%
import numpy as np
from sklearn import datasets
import seaborn as sns

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

#用感知机模型直接训练数据机
#%%
from sklearn.linear_model import Perceptron
ppn=Perceptron(n_iter=40,eta0=0.1,random_state=0)
ppn.fit(X_train_std,y_train)
y_pred=ppn.predict(X_test_std)
print('Misclassfied samples: %d' %(y_test!=y_pred).sum())
#%%
#metrics包含许多评价指标
from sklearn.metrics import accuracy_score
print('Accuracy:%.2f' %accuracy_score(y_test,y_pred))

#%%
#绘图函数
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
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
plot_decision_regions(X=X_combined_std,y=y_combined,classifier=ppn,test_idx=range(105,150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
X_combined_std

np.array([5,6,3,4,5,3]).reshape(6,1)

