#透视表
import numpy as np 
import pandas as pd
from pandas import DataFrame,Series

#交叉表是一种用于计算分组频率的特殊透视表
#%%

filepath='C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch09Data Aggregation and Group Operations/data'
tips=pd.read_csv(filepath+'/tips.csv')

# 添加“消费占总额百分比”的列
tips['tip_pct']=tips['tip']/tips['total_bill']


#%%
pd.crosstab(index=[tips.time,tips.day],columns=tips.smoker,margins=True)

