import pandas as pd
from pandas import DataFrame
#%%
csv_data=pd.read_csv('C:/Users/xg302/git/python-data-mining-and-analysis/preforml/data/csv_data.csv')
print(csv_data)

#统计每一列的缺失值
csv_data.isnull().sum()
csv_data.values

'''
消除带有缺失值的特征或样本
'''
csv_data.dropna()  #去除带有na值的样本
csv_data.dropna(axis=1)#去除带有na值的特征
csv_data.dropna(how='all') #去掉那些所有值均为NA的行

csv_data.dropna(thresh=4) #去掉那些非缺失值少于4个的行
csv_data.dropna(subset=['C']) #去掉指定特征出现缺失值的行


'''
插入法改写缺失值:
sklearn.preprocessing 中的Imputer来实现
参数详解:
missing_values:要替换的是values
strategy:替换策略，mean代表平均值 还可以取 median 和 most_frequent
axis:代表采用策略的方式，一般为0，即取每个特征的属性去插入到该特征的缺失值
'''
#%%
from sklearn.preprocessing import Imputer


imr=Imputer(missing_values="NaN",strategy="mean",axis=0)
imr=imr.fit(csv_data)
imputed_data=imr.transform(csv_data.values)
csv_data
imputed_data=pd.DataFrame(imputed_data)
imputed_data

'''
sklearn中的estimator的api:
transformer类的estimator的两个方法:
fit --根据训练集学习模型参数
transform-- 用学习到的参数转换数据

'''

'''
为了保证学习算法能改正确解释有序特征
我们需要将分类型字符传转为整型数值
因此可以映射有序特征
'''
df=pd.DataFrame([['green','M',10.1,'class1'],['red','L',13.5,'class2'],['blue','XL',15.3,'class1']])
df.columns=['color','size','price','classlabel']
df
#%%
size_mapping={
    'XL':3,
    'L':2,
    'M':1
}

df['size']=df['size'].map(size_mapping)
#转换回原来的字符串：反映射字典
# inv_size_mapping={v:k for k ,v in size_mapping.items()} 
inv_size_mapping={v:k for k ,v in size_mapping.items()} 
df['size']=df['size'].map(inv_size_mapping)

'''
对类别进行编码
'''
import numpy as np
#从0开始赋值
class_mapping={label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
type(class_mapping)
df['classlabel']=df['classlabel'].map(class_mapping)
df
inv_class_mapping={v:k for k,v in class_mapping.items()}
df['classlabel']=df['classlabel'].map(inv_class_mapping)


'''
sklearn.LabelEncoder实现类别转换
'''
from sklearn.preprocessing import LabelEncoder

class_le=LabelEncoder()
#fit_transform 是fit和transform的合并
y=class_le.fit_transform(df['classlabel'].values)
y

#通过调用inverse_transform方法得到原始的字符串类型值
class_le.inverse_transform(y)


'''
对离散特征进行独热编码：
即那些特征有多个特征值，但是互相之间没有大小关系时
采用one-hot encoding(ohe)
即一个特征值转变为一个新的特征，是这个特征值的样本该特征主为1
反之为0
使用pandas.get_dummies 来进行ohe
'''
#%%
#get_dummies默认对
#
# 给定的对象中所有字符串类型的列进行ohe转换
pd.get_dummies(df)

