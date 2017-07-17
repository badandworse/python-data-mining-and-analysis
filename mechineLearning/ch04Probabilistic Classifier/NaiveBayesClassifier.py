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
group1.size()
type(group1.size())
group1.size().sum()
# 算出取值Y的各个概率
Y_pro=group1.size()/group1.size().sum()
Y_pro

# 选取出指定行来算取先验概率
dt1=data[['X1','Y']]

group2=dt1.groupby(['Y','X1'])
