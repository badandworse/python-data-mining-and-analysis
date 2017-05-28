#pandas的数据结构介绍
from pandas import Series,DataFrame
import pandas as pd
import numpy as np

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


#%%
#DataFrame
#可看成由多个Series组成的字典,共用一个索引

data={'state':['Ohio','Ohio','Ohio','Nevada','Nevada'],
      'year':[2000,2001,2002,2001,2002],
      'pop':[1.5,1.7,3.6,2.4,2.9]
}
frame=DataFrame(data)
frame
'''
output:
pop	state	year
0	1.5	Ohio	2000
1	1.7	Ohio	2001
2	3.6	Ohio	2002
3	2.4	Nevada	2001
4	2.9	Nevada	2002
'''

#可以指定列序列
DataFrame(data,columns=['year','state','pop'])

#传出的列在的数据中找不到，就会产生NA值
frame2=DataFrame(data,columns=['year','state','pop','debt'],index=['one','two','three','four','five'])
frame2
frame2.columns

#通过类似字典的标记的方式或属性方式，可以将DataFrame
#的列获取为一个Series,返回的Series和原DateFrame有相同的索引
frame2['state']
frame2.year
frame2.ix['three']


#列可以通过赋值而进行修改
frame2['debt']=16.5
frame2

#将列表或数组赋值给某个列时
#其长度必须跟DataFrame的长度相匹配
#而赋值的是一个Series时，就会精准匹配DataFrame的索引
#所有的空位都将被填上缺失值
#通过索引方式返回的列只是相应数据的视图而已
frame2['debt']=np.arange(5.)
frame2

#%%
val=Series([-1.2,-1.5,-1.7,8],index=['two','four','five','six'])
frame2['debt']=val
frame2


frame2['eastern']=frame2.state=='Ohio'
frame2
del frame2['eastern']
frame2

frame2.columns

#%%
#嵌套字典，将其传给DataFrame:外层的键作为列，内存键作为行索引
pop={'Nevada':{2001:2.4,2002:2.9},
     'Ohio':{2000:1.5,2001:1.7,2002:3.6}
}
frame3=DataFrame(pop)
frame3
frame3.T

#显式指定索引
DataFrame(pop,index=[2001,2002,2003])

pdata={'Ohio':frame3['Ohio'][:-1],
       'Nevada':frame3['Nevada'][:2]
     }

data2=[1,2,3,45]
DataFrame(data2,index=['1','2','22','33'])

frame3.index.name='year'
frame3.columns.name='state'
frame3

#values已二维ndarray的形式返回DataFrame中的数据
frame3.values

#如果DataFrame各列的数据类型就会选用能兼容所有列的数据类型
frame2.values


#%%
#索引对象 pandas中的索引对象负责管理轴标签
#和其他元数据（比如轴名称）
obj=Series(range(3),index=['a','b','c'])
index=obj.index

index[1:]

#index对象是不可修改的，尝试修改会报错
index[1]='ds'
#输入： Index does not support mutable operations

#%%
#index不可修改使得index对象可以在多个数据结构之间安全共享
index=pd.Index(np.arange(3))
obj2=Series([-1.5,-2.5,0],index=index)
obj2.index is index

#Index 将轴标签表示为一个由Pyhton对象组成的Numpy数组
#Int64Index 针对整数的特殊Index
#MultiIndex ‘层次化’索引对象，表示单轴上的多层索引，可看成由元祖组成的数组
#DatetimeIndex 存储纳秒级时间戳（Numpy的datetime64）
#PeriodIndex 针对Period数据（时间间隔）的特殊Index

frame3
frame3.index
'Ohio' in frame3.columns
'2013' in frame3.index

frame3.index.is_unique

#append  连接另一个Index对象，产生一个新的Index 
#diff    计算差集
#intersection 计算交集 
#union  计算并集
#isin   计算一个指示各值是否都包含在参数集合中的布尔型数组
#delete 删除传入的值，并得到新的Index
#drop   删除传入的值，并得到新的Index
#insert 将元素插入到索引i处，并得到新的Index
#is_monotonic  当各元素均大于前一个元素时，返回True
#is_unique    当Index没有重复值时，返回True  
#unique       计算Index中唯一值的数组


