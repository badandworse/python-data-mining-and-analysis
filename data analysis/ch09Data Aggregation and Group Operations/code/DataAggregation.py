#数据聚合

import numpy as np 
import pandas as pd
from pandas import DataFrame,Series

df=DataFrame({'key1':['a','a','b','b','a'],
              'key2':['one','two','one','two','one'],
              'data1':np.random.randn(5),
              'data2':np.random.randn(5)})
df
grouped=df.groupby('key1')
grouped['data1'].quantile(0.9)

#%%
# 自定义聚合函数
def peak_to_peak(arr):
    return arr.max()-arr.min()
# 然后传入aggregate或agg方法
grouped.agg(peak_to_peak)
df
grouped.describe().T

grouped['data1'].min()

filepath='C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch09Data Aggregation and Group Operations/data'
tips=pd.read_csv(filepath+'/tips.csv')

# 添加“消费占总额百分比”的列
tips['tip_pct']=tips['tip']/tips['total_bill']
tips['tip_pct'][:6]

# 面向列的多函数应用
grouped=tips.groupby(['sex','smoker'])
#  把需要操作的列提取出来
grouped_pct=grouped['tip_pct']
grouped_pct.agg('mean')
#  然后传入聚合函数
grouped_pct.agg(['mean','std',peak_to_peak])

# 自定义列名:传入 (name,function)元祖组成的列表，
# 各元祖的第一个元素就会被用作DataFrame的列名
grouped_pct.agg([('foo','mean'),('bar',np.std)])

#%%
functions=['count','mean','max']
result=grouped['tip_pct','total_bill'].agg(functions)
result
result['tip_pct']

ftuples=[('Durchschnitt','mean'),('Abweichung',np.var)]
grouped['tip_pct','total_bill'].agg(ftuples)

# 对不同的列应用不同的函数
#  向agg传入一个从列名映射到函数的字典
grouped.agg({'tip':np.max,'size':'sum'})

#%%
# 对一个列应用多个函数：字典value传入函数列表
grouped.agg({'tip_pct':['min','max','mean','std'],
             'size':'sum'})

# 以“无索引”的形式返回聚合数据
#  默认时，聚合数据由唯一的分组键组成的索引（可能是层次化的）
#  聚合时传入as_index=False即可禁用该功能
tips.groupby(['sex','smoker'],as_index=False).mean()
tips.groupby(['sex','smoker'],as_index=True).mean()
