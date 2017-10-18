'''
将数据集分割为训练集和测试集
'''
#%%
import pandas as pd
from pandas import DataFrame
import sklearn
import numpy as np

filePath='C:/Users/xg302/git/python-data-mining-and-analysis/preforml/data/'
data_wine=pd.read_csv(filePath+'wine_data.csv')
#%%
data_wine.columns=['Class label','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids',
                    'Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline ']

print('class labels',np.unique(data_wine['Class label']))

'''
利用sklearn.cross_validation进行数据分割
train_test_split:
test_size:参数占给定数据集的比例
'''

#%%
from sklearn.cross_validation import train_test_split

X,y=data_wine.iloc[:,1:].values,data_wine.iloc[:,0].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_s


'''
特征缩放:
统一特征取值范围,使特征都在统一范围内
很多情况下会取得好的效果
'''

#归一化:最小-最大缩放
#%%
from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
X_train_norm=mms.fit_transform(X_train)
x_test_norm=mms.transform(X_test)

'''
标准化:
使用标准化，能将特征值缩放到以0为中心，标准差为1
标准化后的特征形式服从正态分布
sklearn 中 standardScalar类实现列标准化
'''
#%%
from sklearn.preprocessing import StandardScaler

stdsc=StandardScaler()
X_train_std=stdsc.fit_transform(X_train)
X_test_std=stdsc.transform(X_test)


'''
选择有意义的特征:
测试效果很差，说明过拟合，可以通过降低数据的纬度来降低过拟合

'''

'''
L1正则:
权重的绝对值和，
使得最优解很有可能落在坐标轴上,可以减少特征数
L2正则:
权重的平方和
'''
#将penalty取值设置为L1正则即可
from sklearn.linear_model import LogisticRegression

LogisticRegression(penalty='l1')
#%%
lr=LogisticRegression(penalty='l1',C=0.1)
lr.fit(X_train_std,y_train)

print('Training accuracy:',lr.score(X_train_std,y_train))

print('Test accuracy:',lr.score(X_test_std,y_test))

#只有3个值，说明lr用的one-vs-reset
#有3个模型
lr.intercept_

lr.coef_

'''
通过画出正则以后，可以发现，C<0.1时，即正则项系数很大是，正则威力是巨大的
使得所有特征权重都为0
'''

#%%
import matplotlib.pyplot as plt
sns.set()
fig=plt.figure()
ax=plt.subplot(111)
colors=['blue','green','red','cyan','magenta','yellow','black','pink','lightgreen','lightblue']
weights,params=[],[]
for c in [-4, -3, -2, -1,  0,  1,  2,  3,  4,  5]:
    lr=LogisticRegression(penalty='l1',C=10**c,random_state=0)
    lr.fit(X_train_std,y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)


weights=np.array(weights)
for column,color in zip(range(weights.shape[1]),colors):
    plt.plot(params,weights[:,columns],label=data_wine.columns[column+1],color=color)

plt.axhline(0,color='black',linestyle='--',linewidth=3)
plt.xlim([10**(-5),10**5])
plt.ylabel('weight cofficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',bbox_to_anchor=(1.38,1.03),ncol=1,fancybox=True)
plt.show() 