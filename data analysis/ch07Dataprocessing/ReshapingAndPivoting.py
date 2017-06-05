import pandas as pd
import numpy as np
from pandas import DataFrame,Series

#重塑层次化索引
data=DataFrame(np.arange(6).reshape((2,3)),index=pd.Index(['Ohio','Colorado'],name='state'),columns=pd.Index(['one','two','three'],name='number'))
data

# stack:将数据列‘旋转’为行
# ustack:将数据的行‘旋转’为列
# 默认情况下，两者都是操作最内层
# is will return True if two variables point to the same object
data.stack()
result=data.stack()
result.unstack()
# 为0则是最外层的
result.unstack(0)
# 如果不是所有的级别只都能在个分组中找到的话，unstack会引入缺失数据
# stack则默认滤除缺失数据
s1=Series([0,1,2,3],index=['a','b','c','d'])
s2=Series([4,5,6],index=['c','d','e'])
data2=pd.concat([s1,s2],keys=['one','two'])
data2
data2.unstack()
data2.unstack().stack() 

data2.unstack().stack(dropna=False) 

df=DataFrame({'left':result,'right':result+5},columns=pd.Index(['left','right'],name='side'))
df

df.unstack('state')
df.unstack('state').stack('side')



#  True
type(data) is DataFrame
#  False
data is DataFrame

#%%
#将‘长格式’旋转为‘宽格式’
filepath2='C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch07Dataprocessing/'
data=pd.read_csv(filepath2+'macrodata.csv')
# 先选出需要的列
periods=pd.PeriodIndex(year=data.year,quarter=data.quarter,name='date')

data.to_records()
data=DataFrame(data.to_records(),columns=pd.Index(['realgdp','infl','unemp'],name='item'),index=periods.to_timestamp('D','end'))
data.index
data[:10]

# reset_index()将层次索引分解为列，同时返回为默认索引
ldata=data.stack().reset_index().rename(columns={0:'value'})

ldata[:10]
# pivot的3个参数中，前两个参数值分别用作行和列索引，
# 最后一个参数则是用于填充DataFrame的数据列的列名
ldata.head()
ldata['value']
pivoted=ldata.pivot('date','item','value')

# 如果忽略最后一个参数，列就会编程一个层次化的，外层为'value'
ldata.pivot('date','item')

pivoted.head()
pivoted

ldata['value2']=np.random.randn(len(ldata))
ldata[:10]
pivoted=ldata.pivot('date','item')
pivoted[:5]
pivoted['value'][:5]
ldata.head()
# 先建立层次索引
ldata.set_index(['date','item'])
# 在将item行转换为列
ldata.set_index(['date','item']).unstack('item')