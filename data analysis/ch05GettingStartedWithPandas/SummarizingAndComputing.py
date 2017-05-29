import pandas as pd
from pandas import DataFrame,Series
import numpy as np

df =DataFrame([[1.4,np.nan],[7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]],index=['a','b','c','d'],columns=['one','two'])
df
#sum默认返回各列和的Series,NA值会自动排除
df.sum()

#axis=1将会对行进行求和运算
df.sum(axis=1)

#返回每列的平均值，axis=1时为返回行的平均值
#skipna 排除缺失值，默认为True
#level 轴是层次话索引的，则根据level分组约简
df.mean(skipna=False)

#默认返回每列中最大值项的索引（即行编号）
df.idxmax()
#默认返回每列中最小值项的索引（即行编号）
df.idxmin()

#cumsum返回累积和，默认为列
df.cumsum()
df.cumsum(axis=1)
#当前列及前面的各列中最大值，自动跳过NA
df.cummax()
#对数据进行统计汇总
df.describe()

df.mad()
#var方差
df.var()
#三阶矩=E[((X-mean)/std)^3] 它的正负及大小用来衡量分布的不对称性

#四阶矩 它用来描述随机变量分布的峰态。
df.kurt()
#一阶差分，每列元素与前一个元素的差，没有或者无效值就为无效值
df.diff()
df

#%%
#相关系数与协方差
from pandas_datareader import data as web
from pandas_datareader.data import Options
import pandas_datareader as pdr
import datetime
'''all_data={}
start=datetime.datetime(2010,1,1)
end=datetime.datetime(2013,1,27)
for ticker in ['AAPL','IBM','MSFT','GOOG']:
    all_data[ticker]=web.get_data_yahoo(ticker,start,end)

all_data'''
print('fxxk')

start=datetime.datetime(2010,1,1)
end=datetime.datetime(2013,1,27)
inflation = web.DataReader(["CPIAUCSL", "CPILFESL"], "fred", start, end)
inflation.head()
#价格的百分数变化
returns=inflation.pct_change()
returns.tail()

#计算协方差
returns.CPIAUCSL.corr(returns.CPILFESL)

returns.corr()
#相关系数
returns.CPIAUCSL.cov(returns.CPILFESL)
returns.cov()

#DataFrame.corrwith 计算其列或行跟另一个Series的相关系数
#如果传入一个DataFrame，则会计算按列名匹配的相关系数
#相关系数Cov(x,y)/(D(X)D(Y))^(1/2)  判断两个变量是否有关系
#cov(x,y)=E((X-E(X)(Y-E(Y))))    判断两个变量是否是同方向变化
returns.corrwith(returns.CPILFESL)
returns.corrwith(returns)

#%%
#唯一值、值计数以及成员资格
obj=Series(['c','a','d','a','a','b','b','c','c'])
#uniques去掉重复值
uniques=obj.unique()
uniques
#计算每个元素在series中出现的次数,按出现次数降序排列
obj.value_counts()

#pandas中有value_counts. sort=False, 按出现先后顺序
pd.value_counts(obj.values,sort=False)

#isin 计算一个表示‘Series各值是否包含于传入的值序列中’的布尔型数组
mask=obj.isin(['b','c'])
mask
obj[mask]

data=DataFrame({'Qu1':[1,2,3,4,3],
                'Qu2':[2,3,1,2,3],
                'Qu3':[1,5,2,4,4]
})

data

result=data.apply(pd.value_counts).fillna(0)
result

#%%
#处理缺失数据
string_data=Series(['aardvark','artichoke',np.nan,'avocado'])
#python内置的None值也会被当作NA处理
string_data[0]=None
string_data.isnull()

#NA处理方法：
# dropna 根据各标签的值中是否存在缺失数据对轴标签进行过滤，可通过阈值调节对缺失值的容忍度
# fillna 用指定值或插值方法(如fill或bfill)填充缺失数据
# isnull 返回一个含有bool值对象，这些布尔值表示那些值是缺失值（NA）
# notnull isnull的否定式

string_data.dropna()
string_data.fillna(1)

#%%
#滤除缺失数据

from numpy import nan as NA
data=Series([1,NA,3.5,NA,7])
data.dropna()
#通过bool型索引达到这个目的
data[data.notnull()]

data=DataFrame([[1.,6.5,3.],[1.,NA,NA],[NA,NA,NA],[NA,6.5,3.]])
#对于DataFrame dropna默认丢弃任何缺失值的行
cleaned=data.dropna()
data
cleaned
#传入how='all'则是只丢弃全为NA的哪些行
data.dropna(how='all')
#要用这种方式丢弃列，只需传入axis=1即可:
data[4]=NA
data
data.dropna(axis=1,how='all')

df=DataFrame(np.random.randn(7,3))
df.ix[:4,1]=NA
df
df.ix[:2,2]=NA
df
df.dropna(thresh=3)

#%%

#fillna函数的参数：
# value 填充缺失值的标量值或字典对象 
# method 插值方式，如果函数调用时未指定其他参数的话，默认为‘ffill’
# axis  待填充的轴，默认axis=0
# inplace 修改调用者对象而不产生副本
# limit   （对于前向和后向的填充）可以连续填充的最大数量


#填充确实数据
df.fillna(0)

df.fillna({1:0.5,2:-1})

#fillna默认返回新对象，但也可以对现有对象进行修改:
df.fillna(0,inplace=True)
df

df=DataFrame(np.random.randn(6,3))
df.ix[2:,1]=NA;df.ix[4:,2]=NA
df
df.fillna(method='ffill')
#每列最多填入两个缺省值
df.fillna(method='ffill',limit=2)

#每列缺省值的填充值为该列的平均值
df.fillna(df.mean())

