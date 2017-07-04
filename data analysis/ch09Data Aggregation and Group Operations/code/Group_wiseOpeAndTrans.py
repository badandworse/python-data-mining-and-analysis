#分组级运算和转换
import numpy as np 
import pandas as pd
from pandas import DataFrame,Series


df=DataFrame({'key1':['a','a','b','b','a'],
              'key2':['one','two','one','two','one'],
              'data1':np.random.randn(5),
              'data2':np.random.randn(5)})


# add_prefix()在每一列名前加上指定字符串
k1_means=df.groupby('key1').mean().add_prefix('mean_')
k1_means


pd.merge(df,k1_means,left_on='key1',right_index=True)
#%%
people=DataFrame(np.random.randn(5,5),
                 columns=['a','b','c','d','e'],
                 index=['Joe','Steve','Wes','Jim','Travis'])

people
key=['one','two','one','two','one']
people.groupby(key).mean()
# transform会将一个函数应用到各个分组，
# 然后将结果放置到适当的位置上。
people.groupby(key).transform(np.mean)

#%%
def demean(arr):
    return arr-arr.mean



#apply 一般性的“拆分-应用-合并”
# agg和transfrom都要求传入的函数只能产生两种结果:
#  一个可以广播的标量值，或一个相同大小的结果数组


#%%
# 根据分组选出最高的5个tip_pct
def top(df,n=5,column='tip_pct'):
    return df.sort_index(by=column)[-n:]

top(tips,n=6)
tips.groupby('smoker').apply(top)

# 传入apply函数接受其他参数或关键字,
# 则可以将这些内容放在函数后面一并传入
tips.groupby(['smoker','day']).apply(top,n=1,column='total_bill')

result=tips.groupby('smoker')['tip_pct'].describe()

f=lambda x :x.describe()
tips.groupby('smoker')['tip_pct'].apply(f)

#禁止分组键
# 分组建会跟原始对象的索引共同构成结果对象中的层次化索引
# 将group_keys=False传入groupby即可禁止该效果
tips.groupby('smoker',group_keys=False).apply(top)

#%%
#分位数和桶分析
frame=DataFrame({'data1':np.random.randn(1000),
                 'data2':np.random.randn(1000)})

frame['data1'][:10]
# 将frame.data1分为4个面元，元素值显示当前元素在哪个面元
# cut只给定面元数量，则给定的是等长面元
factor=pd.cut(frame.data1,4)
factor[:10]

# cut返回的对象可以直接用于gourpby

#%%
def get_statas(group):
    return {'min':group.min(),'max':group.max(),
            'count':group.count(),'mean':group.mean()}

grouped=frame.data2.groupby(factor)
grouped.apply(get_statas).unstack()

# qcut返回的等数据量的面元
grouping=pd.qcut(frame.data1,10,labels=False)
grouped=frame.data2.groupby(grouping)
grouped.apply(get_statas).unstack()

#%%
#示例 用特定于分组的值填充缺失值
s=Series(np.random.randn(6))
s[::2]=np.nan
s
s.fillna(s.mean())

# 不同分组填充不同的值，数据分组，
# 使用apply和一个能够对数据库调用fillna的函数即可
#%%
states=['Ohio','New York','Vermont','Florida',
         'Oregon','Nevada','California','Idaho']

group_key=['East']*4+['West']*4
data=Series(np.random.randn(8),index=states)
data[['Vermont','Nevada','Idaho']]=np.nan
data

data.groupby(group_key).mean()

fill_mean=lambda g:g.fillna(g.mean())
data.groupby(group_key).apply(fill_mean)

# 预设填充值
fill_values={'East':0.5,'West':-1}

fill_func=lambda g:g.fillna(fill_values[g.name])
data.groupby(group_key).apply(fill_func)

#示例：随机采样和排列

#%%
# 构建英语型扑克牌
suits=['H','S','C','D']
card_val=(list(range(1,11))+[10]*3)*4
card_val
base_names=['A']+list(range(2,11))  +['J','K','Q']
cards=[]
for suit in suits:
    cards.extend(str(num)+suit for num in base_names)

deck =Series(card_val,index=cards)
deck

#%%
def draw(deck,n=5):
    return deck.take(np.random.permutation(len(deck))[:n])


draw(deck,n=5)
#%%
get_suit=lambda card:card[-1]
deck.groupby(get_suit).apply(draw,n=2)
deck.groupby(get_suit,group_keys=False).apply(draw,n=2)


#示例：分组加权平均数和相关系数
#%%
df=DataFrame({'category':['a','a','a','a','b','b','b','b'],
              'data':np.random.randn(8),
              'weights':np.random.rand(8)})

df

#%%
grouped=df.groupby('category')
get_wavg=lambda g:np.average(g['data'],weights=g['weights'])
grouped.apply(get_wavg)

#%%
filepath='C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch09Data Aggregation and Group Operations/data'

close_px=pd.read_csv(filepath+'/stock_px.csv',parse_dates=True,index_col=0)

close_px.info()
close_px[-4:]

#%%
rets=close_px.pct_change().dropna()
spx_corr=lambda x : x.corrwith(x['SPX'])
by_year=rets.groupby(lambda x : x.year)
by_year.apply(spx_corr)

by_year['AAPL'].corr(by_year['MSFT'])

#%%
# 列与列之间的相关系数
by_year.apply(lambda g:g['AAPL'].corr(g['MSFT']))


#示例:面向分组的线性回归
#%%
import statsmodels.api as sm
def regress(data,yvar,xvars):
    Y=data[yvar]
    X=data[xvars]
    X['intercept']=1.
    result=sm.OLS(Y,X).fit()
    return result.params

by_year.apply(regress,'AAPL',['SPX'])
