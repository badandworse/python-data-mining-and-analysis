#层次化索引
#%%
import pandas as pd
import numpy as np
from pandas import DataFrame, Series

data=Series(np.random.randn(10),index=[['a','a','a','b','b','b','c','c','d','d'],[1,2,3,1,2,3,1,2,2,3]])
data
data.index
data['b']
data['b':'c']
data.ix[['b','d']]

data[:,2]

data.unstack()

data.unstack().stack()
frame=DataFrame(np.arange(12).reshape((4,3)),index=[['a','a','b','b'],[1,2,1,2]],columns=[['Ohio','Ohio','Colorado'],['Green','Red','Green']])
frame

frame.index.names=['key1','key2']
frame.columns.names=['state','color']
frame
#分布索引可以使得选取数据非常方便
frame['Ohio']


#%%
#重新分级排序
frame
#swaplevel 接受两个级别编号或名称，并返回一个互换了级别的新对象
frame.swaplevel('key1','key2')

#sortlevel 则根据单个级别中的值对数据进行排序。
frame.sortlevel(1)
frame.sortlevel()


#%%
#根据基本汇总统计
frame.sum(level='key2')
frame.sum(level='color',axis=1)

#%%
#使用DataFrame的列

frame=DataFrame({'a':range(7),'b':range(7,0,-1),'c':['one','one','one','two','two','two','two'],
                 'd':[0,1,2,0,1,2,3]
                                    })
frame
frame.index

#DataFrame的set_index函数将其一个或多个列转换为行索引，并创建一个新的DataFrame
# 默认情况将这些列从DataFrame中移除，但也可以将其保留下来
frame2=frame.set_index(['c','d'])
frame2
frame.set_index(['c','d'],drop=False)
#reset_index的功能跟set_index刚好相反.层次化索引的级别会被转移到列里面
frame2.reset_index()
