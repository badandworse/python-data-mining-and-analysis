'''
随机森林算法：
通过自助法构建大小为n的一个训练集，即重复抽样选择n个训练样例
对于刚得到的训练集，构建决策树
同时对每个节点:
通过不重复抽样选择d个特征
利用上面的d个特征，选择某种度量分割节点 
重复步骤1.2 k次
对于每一个测试样例，对k颗决策树的预测结果进行投票，
票数最多的结果就是随机森林的预测结果


'''


#%%
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

#数据处理
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

#%%
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))



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
 

#随机森林
#%%
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(criterion="entropy",n_estimators=10,random_state=1,n_jobs=2)
forest.fit(X_train_std,y_train)

plot_decision_regions(X=X_combined_std,y=y_combined,classifier=forest,test_idx=range(105,150))

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()