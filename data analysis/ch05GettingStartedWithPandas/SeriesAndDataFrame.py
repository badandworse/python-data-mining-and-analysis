#pandas的数据结构介绍
from pandas import Series,DataFrame
import pandas as pd

#%%
#Series
obj=Series([4,7,-5,3])
obj

obj.values
obj.index

#创建带指定索引的Series
obj2=Series([4,7,-5,3],index=['d','b','a','c'])
obj2
#与普通的Numpy数组相比，你可以通过索引的方式选取Series的单个或者一组值
obj2.index
obj2[['c','d','d']]

'b' in obj2
obj2[obj2>2]

#如果数据直接存放在dict中，可直接创建Series
sdata={'Ohio':3500,'Texas':71000,'Oregon':16000,'Utah':5000}
obj3=Series(sdata)
obj3

#sdata跟states索引相匹配的那3个值会被找出来并放到相应的位置上
#'California'所对应的sdata值找不到，所以其结果就为NaN
states=['California','Ohio','Oregon','Texas']
obj4=Series(sdata,index=states)
obj4

#pandas的isnull和notnull函数可用于检测缺失数据
pd.isnull(obj4)
pd.notnull(obj4)

#Series的功能：在算术运算中会自动对齐不同的索引的数据
obj3,obj4
obj3+obj4


obj4.name='population'
obj4.index.name='state'
obj4

#索引可以通过复制的方式直接修改
obj.index=['Bob','Steve','Jeff','Ryan']
obj