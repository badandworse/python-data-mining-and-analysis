#%%
import numpy as np 
import pandas as pd
from pandas import DataFrame,Series

#%%
data1=[6,5,3]
arr1=np.array(data1)
arr1

print (1)

#%%
df=DataFrame({'key1':['a','a','b','b','a'],
              'key2':['one','two','one','two','one'],
              'data1':np.random.randn(5),
              'data2':np.random.randn(5)})

df

grouped=df['data1'].groupby(df['key1'])
grouped
grouped.mean()

means=df['data1'].groupby([df['key1'],df['key2']]).mean()
means

means.unstack()

#%%
# 分组键可以是任何长度适当的数组
states=np.array(['Ohio','California','California','Ohio','Ohio'])
years=np.array([2005,2005,2006,2005,2006])
df['data1'].groupby([states,years]).mean()

# 列名也可以做为分组键
df.groupby(['key1','key2']).mean()

# GroupBy.size 返回一个含有分组大小的seirse
df.groupby(['key1','key2']).size()

#  groupby 分组求平均数时，对于非数值列，
#  或缺失值都会自动排除

# 对分组进行迭代
#  GroupBy对象支持迭代，
#  可以产生一组二元元祖（有分组名和数据块组成）
#%%
for name ,group in df.groupby('key1'):
    print('name:',name)
    print(group)

df

# 对于多重键，元祖的第一个元素是由键值组成的元祖
#%%
for (k1,k2),group in df.groupby(['key1','key2']):
    print(k1,k2)
    print(group)


list(df.groupby('key1'))
pieces=dict(list(df.groupby('key1')))
pieces

# groupby 默认在axis=0上分组，可以修改参数在axis=1上分组
df.dtypes

grouped=df.groupby(df.dtypes,axis=1)
dict(list(grouped))

#选取一个或一组列
#如果用一个（单个字符串）或一组（字符串数组）
#列名对齐进行索引，能实现选取部分列进行聚合
df.groupby('key1')['data1'].mean()
df.groupby(['key1','key2'])[['data2']].mean()

#%%
#这种索引操作返回一个已分组的DataFrame或已分组的Series
s_grouped=df.groupby(['key1','key2'])['data2']
s_grouped
s_grouped.mean()

#%%
#通过字典或Series进行分组
people=DataFrame(np.random.randn(5,5),
                 columns=['a','b','c','d','e'],
                 index=['Joe','Steve','Wes','Jim','Travis'])


people.ix[2:3,['b','c']]=np.nan
people
# 利用字典来分类
mapping ={'a':'red','b':'red','c':'blue',
          'd':'blue','e':'red','f':'orange'}
by_column=people.groupby(mapping,axis=1)          
by_column.sum()

# Series也可以到达相同目的
map_series=Series(mapping)
map_series
people.groupby(map_series,axis=1).count()


#通过函数进行分组: 
# 任何被当作分组键的函数都会在各
# 个索引值被调用一次

# 根据人名长度进行分组
# 传入len函数即可
people.groupby(len).sum()
people

# 函数可以和数组、列表、字典、Series混用也不是问题
key_list=['one','one','one','two','two']
people.groupby([len,key_list]).min()


#根据索引级别分组
# 层次化索引数据集可以根据索引级别进行聚合
# 通过level关键之传入级别编号或名称即可
columns=pd.MultiIndex.from_arrays([['US','US','US','JP','JP'],[1,3,5,1,3]],names=['city','tenor'])

#%%
hier_df=DataFrame(np.random.randn(4,5),
                  columns=columns)
hier_df

grouped=hier_df.groupby(level='city',axis=1)
grouped.count()

