import pandas as pd
import numpy as np
from pandas import DataFrame,Series

#%%
#重新索引
obj=Series([4.5,7.2,-5.3,3.6],index=['d','b','a','c'])
obj
#reindex 将会根据新索引进行重排，
#如果某个索引值不存在，就引入缺失值
obj.reindex(['a','b','c','d','e'])

obj.reindex(['a','b','c','d','e'],fill_value=0)

obj3=Series(['blue','purple','yellow'],index=[0,2,4])
obj3
#method 使用ffill可以实现前向值填充, reindex(差值)method选项:
#ffill或pad 前向填充 
#bfill或backfill 后向填充 
obj3.reindex(range(6),method='ffill')

#DataFrame的reinde可以修改行、列索引或同时修改两个，
#仅传入一个则会重新索引行

frame=DataFrame(np.arange(9).reshape((3,3)),index=['a','c','d'],\
    columns=['Ohio','Texas','California'])
frame

frame2=frame.reindex(['a','b','c','d'])
frame2

#columns关键字即可重新索引列
states=['Texas','Utah','California']
frame.reindex(columns=states)


frame.reindex(index=['a','b','c','d'],method='ffill',columns=states)

#利用ix的标签索引功能，重新索引任务可以变得更简洁
frame.ix[['a','b','c','d'],states]

#reindex函数的参数
#method   插值方式，具体参数请参见表5-4
#fill_value 在重新索引的过程中，需要引入缺失值时使用的替代之
#limit    前向或后向填充时的最大填充量
#level    在MultiIndex的指定级别上匹配简单索引，否则选取其子集
#copy     默认为True，无论如何都复制，如果为False，则新旧相等就不复制


#%%
#丢弃指定轴上的项,drop 参数为一个索引数组或列表
obj=Series(np.arange(5.),index=['a','b','c','d','e'])
new_obj=obj.drop('c')
obj
new_obj
obj.drop(['d','c'])

data=DataFrame(np.arange(16).reshape((4,4)),index=['Ohio','Colorado','Utah','New York'],columns=['one','two','three','four'])
data
data.drop(['Colorado','Utah'])
#axis默认值为0
data.drop('two',axis=1)
data.drop(['two','one'],axis=1

#%%
#索引、选取和过滤
obj=Series(np.arange(4.),index=['a','b','c','d'])
obj
obj['b']
obj[1]
obj[2:4]
obj[['b','a','d']]

#利用标签的切片运算是包含末端的
obj['b':'c']=5
obj

#通过索引获取一个或多个列
data=DataFrame(np.arange(16).reshape((4,4)),index=['Ohio','Colorado','Utah','New York'],columns=['one','two','three','four'])
data
data['two']
#通过切片或bool型数组选取行:
data[:2]
data[data['three']>5]


#使用ix在DataFrame进行标签索引
data.ix['Colorado',['two','three']]
data.ix['Colorado',['two','one','four']]
data.ix[2]
data.ix[:'Utah','two']

data.ix[data.three>5,:2]

#DataFrame的索引项：
#obj[val] 选取DataFrame的单个列或一组列。
# bool型数组（过滤行）、切片（行切片）、布尔型DataFrame（根据条件设置值）
#obj.ix[val] 选取DataFrame的单个行或一组行
#obj.ix[:,val]选取单个列或列子集
#obj.ix[val1,val2] 同时选取行或列
#reindex 将一个或多个轴匹配到新的索引。
# 默认值为行索引，使用columns改列索引
#xs方法  根据标签选取单行或单列，并返回一个Series
#icol、irow方法  根据整数选取单列或单行，并返回一个Series
#get_value 、set_value 根据行标签和列标签选取单个值


#%%
#算术运算和数据对齐
#包含不同索引的两个Series相加时，在不重叠的索引处引入NA值
s1=Series([7.3,-2.5,3.4,1.5],index=['a','c','d','e'])
s2=Series([-2.1,3.6,-1.5,4,3.1],index=['a','c','e','f','g'])
s1+s2
#DataFrame的对齐操作会同时发生在行或列上
df1=DataFrame(np.arange(9.).reshape((3,3)),columns=list('bcd'),index=['Ohio','Texas','Colorado'])
df2=DataFrame(np.arange(12.).reshape((4,3)),columns=list('bde'),index=['Utah','Ohio','Texas','Oregon'])
df1+df2

#%%
#在算术方法中填充值
df1=DataFrame(np.arange(12.).reshape((3,4)),columns=list('abcd'))
df2=DataFrame(np.arange(20).reshape((4,5)),columns=list('abcde'))
df1
df2
df1+df2
#使用add并传出一个相加的DataFrame及一个fill_value参数
#其他类似的方法包括 sub(-),div(/),mul(*)
df1.add(df2,fill_value=0)
df1.reindex(columns=df2.columns,fill_value=0)

#%%
#DataFrame和Series之间的运算
frame=DataFrame(np.arange(12.).reshape((4,3)),columns=list('bde'),index=['Utah','Ohio','Texas','Oregon'])

series1=frame.ix[0]
frame
series1

#默认情况下，DataFrame和Series之间的算术运算会
# 将Series的索引匹配到DataFrame的列，然后向下一直广播
frame-series1

#如果某个索引值在DataFrame列或Series的索引中找不到，
# 则参与运算的两个对象会被重新索引以形成并集
series2=Series(range(3),index=['b','c','f'])
frame+series2

series3=frame['d']
series3
#传入轴号就是希望匹配的轴，本例中，希望匹配DataFrame的行索引并进行广播
frame.sub(series3,axis=0)

#%%
#函数应用和映射
frame=DataFrame(np.random.randn(4,3),columns=list('abe'),index=['Utah','Ohio','Texas','Oregon'])
frame
np.abs(frame)

#将函数应用到各行或各列上，axis的默认值为0
f=lambda x:x.max()-x.min()
frame.apply(f)
frame.apply(f,axis=1)

frame.max()

def f(x):
    return Series([x.min(),x.max()],index=['min','max'])

frame.apply(f)

#如果想运用作用到元素级的函数，则使用applymap：
#而Series中的应用到元素级函数的map为map
formatF=lambda x :'%.2f' %x
frame.applymap(formatF)

frame['e'].map(formatF)


#%%
#排序和排名
#sort_index 返回一个已排序的新对象
obj=Series(range(4),index=['d','a','b','c'])
obj.sort_index()
#对于DataFrame，则可以根据任意一个轴上的索引进行排序,axis默认值为0
frame=DataFrame(np.arange(8).reshape((2,4)),index=['three','one'],columns=['d','a','b','c'])
frame.sort_index()
frame.sort_index(axis=1)
frame.sort_index(axis=1,ascending=False)

#若要按值对Series进行排序，可使用其order方法
obj=Series([4,7,-3,2])
obj.order()

#在排序时，任何缺失值默认都会被放到Series的末尾
obj=Series([4,np.nan,7,np.nan,-3,2])
obj.order()

#DataFrame排序时，将一个或多个列的名字传递给by选项，
# 即可根据一个或多个列中的值进行排序
frame=DataFrame({'b':[4,7,-3,2],'a':[0,1,0,1]})
frame
frame.sort_index(by='b')
frame.sort_index(by=['a','b'])

#排名，默认情况是相等的分组中，为各个值分配平均排名，即method='average'
#min 使用整个分组的最小排名
#max 使用整个分组的最大排名
#first 按值在原始数据中的出现顺序排名
obj=Series([7,-5,7,4,2,0,4])
obj
obj.rank()
obj.rank(method='first')
obj.rank(method='min')
obj.rank(method='max')

#降序进行排序
obj.rank(ascending=True,method='max')

#DataFrame可以在行或列上计算排名
