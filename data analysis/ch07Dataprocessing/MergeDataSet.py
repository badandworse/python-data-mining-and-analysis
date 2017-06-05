import pandas as pd
from pandas import DataFrame,Series
import numpy as np

#数据库风格的DataFRAME合并
# 数据库的合并(merge)或连接(join)运算是通过一个或多个键将行
# 链接起来的
df1=DataFrame({'key':['b','b','a','c','a','a','b'],'data1':range(7)})
df2=DataFrame({'key':['a','b','d'],'data2':range(3)})
df1
df2

# pd中的merge就会默认将重叠列的列名当作键,最后给on属性赋值显示指定
# df1和df2的重叠列为key
pd.merge(df1,df2)
pd.merge(df1,df2,on='key')


# 两列对象的列名不同，可以分别指定
df3=DataFrame({'lkey':['b','b','a','c','a','a','b'],'data1':range(7)})
df4=DataFrame({'rkey':['a','b','d'],'data2':range(3)})
# 默认情况下，merge做的是'inner'链接，结果中的建是交集
pd.merge(df3,df4,left_on='lkey',right_on='rkey')

# 也可使用外联系，外接式则是求并集
pd.merge(df1,df2,how='outer')

# 多对多的合并操作非常简单
df1=DataFrame({'key':['b','b','a','c','a','b'],'data1':range(6)})
df2=DataFrame({'key':['a','b','a','b','d'],'data2':range(5)})
df1
df2
# how='left'代表以左边的为标准做笛卡尔集，而为'right'则是右边
pd.merge(df1,df2,on='key',how='left')
pd.merge(df1,df2,on='key',how='right')

left=DataFrame()


left=DataFrame({'key1':['foo','foo','bar'],
                'key2':['one','two','one'],
                'lval':[1,2,3]})

left
right=DataFrame({'key1':['foo','foo','bar','bar'],
                 'key2':['one','one','one','two'],
                 'rval':[4,5,6,7]})
right

pd.merge(left,right,on=['key1','key2'],how='outer')


pd.merge(left,right,on='key1')

# 合并运算时重复列名处理,suffixes参数赋予字符串元祖，用于追加
# 到重叠名列尾, 默认为suffixes=('_left', '_right')
pd.merge(left,right,on='key1',suffixes=('_left', '_right'))

#%%
#索引上的合并
left1=DataFrame({'key':['a','b','a','a','b','c'],'value':range(6)})

right1=DataFrame({'group_val':[3.5,7]},index=['a','b'])
left1
right1

# 传入left_index=True 或right_index=True 说明索引应该被用来作链接建
pd.merge(left1,right1,left_on='key',right_index=True)

pd.merge(left1,right1,left_on='key',right_index=True,how='outer')

lefth=DataFrame({'key1':['Ohio','Ohio','Ohio','Nevada','Nevada'],
                 'ket2':[2000,2001,2002,2001,2002],
                 'data':np.arange(5.)})

righth=DataFrame(np.arange(12).reshape((6,2)),index=[['Nevada','Nevada','Ohio','Ohio','Ohio','Ohio'],
                 [2001,2000,2000,2000,2001,2002]],columns=['event1','event2'])

righth
lefth
# 对于层次化索引合并，需要以列表的形式指明用作合并的多个列(注意对重复值的索引值处理)
pd.merge(lefth,righth,left_on=['key1','ket2'],right_index=True)

# DataFrame 的join 合并多个带有相同或相似索引的DataFrame对象不管它们之间有没有重叠的列
#　可以传入一组ＤataFRAME来合并

#%%
#轴向链接
arr=np.arange(12).reshape((3,4))
arr
np.concatenate([arr,arr],axis=0)
np.concatenate([arr,arr],axis=1)

# pandas的concat函数用法：
#  对于series，直接将series值和索引粘合在一起:
s1=Series([0,1],index=['a','b'])
s2=Series([0,1,2],index=['a','c','b'])
s3=Series([5,6],index=['f','g'])
pd.concat([s1,s2,s3])
#  默认是在axis=0上工作，最终产生一个新的Series 
#  如果传入axis=1，结果就会变成一个新的DataFrame axis=1为列
#  所谓的行合并是增加行，列合并则是可能增加列
pd.concat([s1,s2,s3],axis=1)
s4=pd.concat([s1*5,s3])
s4

pd.concat([s1,s4],axis=1)

# 传入join='inner'就是求交集
pd.concat([s1,s4],axis=1,join='inner')

# 传入join_axes指定要在其他轴上使用的索引
pd.concat([s1,s4],axis=1,join_axes=[['a','c','b','e']])

# 传入keys参数即可创建层次化索引
result=pd.concat([s1,s2,s3],keys=['one','two','three'])
result

# 沿着axis=1合并，则keys就会变成DataFRAME的列头:
pd.concat([s1,s2,s3],axis=1,keys=['one','two','three'])

# dataFrame同理
df1=DataFrame(np.arange(6).reshape(3,2),index=['a','b','c'],columns=['one','two'])
df2=DataFrame(5+np.arange(4).reshape(2,2),index=['a','c'],columns=['three','four'])
pd.concat([df1,df2],axis=1,keys=['level1','level2'])

# 如果传入的不是列表次而是一个字典，
# 则字典的键就会被当作keys选项的值
pd.concat({'level1':df1,'level2':df2},axis=1)

# 如果设置了keys或levels 可以传入names参数为这两个建立索引
pd.concat([df1,df2],axis=1,keys=['level1','level2'],names=['upper','lower'])

# 忽略分析工作无关的DataFrame行索引
df1=DataFrame(np.random.randn(3,4),columns=['a','b','c','d'])
df2=DataFrame(np.random.randn(2,3),columns=['b','d','a'])

df1
df2
pd.concat([df1,df2],ignore_index=True)

#%%
#合并重叠数据
a=Series([np.nan,2.5,np.nan,3.5,4.5,np.nan],index=['f','e','d','c','b','a'])
b=Series(np.arange(len(a),dtype=np.float64),['f','e','d','c','b','a'])

b[-1]=np.nan
a
b

# where类似于3元运算，
# 根据第一个参数的条件，决定返回值
# 为真就返回第一个，为假就返回第二个
# combine_first与where类似，比较，返回第一个不是non_value的值
np.where(pd.isnull(a),b,a)
b[:-2].combine_first(a[2:])
