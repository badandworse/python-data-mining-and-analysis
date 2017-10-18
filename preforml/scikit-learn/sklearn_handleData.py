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
选择有意义的特征
'''