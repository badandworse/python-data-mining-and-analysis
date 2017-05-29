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


#其他相关pandas的话题

#%%
# 整数索引
ser=Series(np.arange(3.))
ser
#会报错，当series的index为整数时，无法使用这种
ser[-1]
#必须使用存在的索引
ser[1]

#而对于非整数索引，就没有这样的歧义
ser2=Series(np.arange(3.),index=['a','b','c'])
ser2[-1]

#为了保持良好的一致性，如果你得轴索引包含索引器，那么根据整数索引进行数据选取的操作总是面向标签的。
ser.ix[:0]

#可靠的继续位置的索引的方法：
#  Series的iget_value 
#  DataFrame的irow和icol irow 选取每列中对应位置的元素 icol选取每行中对应位置的元素
ser3=Series(range(3),index=[-5,1,3])
ser3.iget_value(2)
frame3=DataFrame(np.arange(6).reshape(3,2),index=[2,0,1])
frame3.irow(0)
frame3.icol(0)

#面板数据
# 没看