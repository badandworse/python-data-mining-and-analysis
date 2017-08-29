#%%
import numpy as np
import pandas as pd
from pandas import DataFrame,Series

#%%
data=DataFrame({'X1':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
                'X2':['S','M','M','S','S','S','M','M','L','L','L','M','M','L','L'],
                'Y':[-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]})

data

group1=data.groupby(['Y'])
# 数据类型为数组
group1.size().index
type(group1.size())
group1.size().sum()
# 算出取值Y的各个概率
Y_pro=group1.size()/group1.size().sum()
Y_pro

# 选取出指定列来算取先验概率
#  先选出'X1'和'Y'
dt1=data[['X1','Y']]

group2=dt1.groupby(['Y','X1'])


s1=group2.size()

type(group2.size())
s_1=s1[1]
s_1
s_1_pro=s_1/s_1.sum()
s_1_pro

s_2=s1[-1]
s_2_pro=s_2/s_2.sum()
s_2_pro


#%%
#  先选出'X2'和'Y'
dt2=data[['X2','Y']]
group3=dt2.groupby(['Y','X2'])
s2=group3.size()
s2
#%%
s_2_1=s2[1]

s_2_1_pro=s_2_1/s_2_1.sum()


#%%
s_2_2=s2[-1]
s_2_2
s_2_2_pro=s_2_2/s_2_2.sum()
s_2_2_pro

#%%
# 对于(2,S),Y=1
p1=Y_pro[1]*s_1_pro[2]*s_2_1_pro['S']
p1


#%%
# 对于(2,S),Y=-1
p2=Y_pro[-1]*s_2_pro[2]*s_2_2_pro['S']
p2
s_2_2_pro['S']
s_2_1_pro[2]
Y_pro[-1]

result=-1 if p1<p2 else 1
result
