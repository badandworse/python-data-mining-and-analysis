#pandas中的绘图函数

#%%
import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import matplotlib.pyplot as plt

# 线性图
#  randn() 返回正态分布随机数
s=Series(np.random.randn(10).cumsum(),index=np.arange(0,100,10))
s.plot()

#柱状图有一个非常不错的用法；
#利用value_counts图形化显示 Series中各值的出现频率，
#比如s.value_counts().plot(kind='bar')
s.value_counts().plot(kind='bar')

#  DataFrame的plot方法会在一个subplot中为各列绘制一条线，
#  并自动创建图例
#  %%
df=DataFrame(np.random.randn(10,4).cumsum(0),columns=['A','B','C','D'],index=np.arange(0,100,10))
#  DataFrame.plot 参数：
# subplots=True 将各个DateFrame列绘制到单独的subplot中
df.plot(subplots=True,sharex=True,sharey=True)

# 柱状图
#  在生成现行图中加入kind='bar'（垂直柱状图）
#  或kind='barh'（水平柱状图）即可生成柱状图
#%%
fig,axes=plt.subplots(2,1)
data=Series(np.random.randn(16),index=list('abcdefghijklmnop'))
data.plot(kind='bar',ax=axes[0],color='k',alpha=0.7)
data.plot(kind='barh',ax=axes[1],color='k',alpha=0.7)
fig

#%%
#DataFrame 柱状图会将每一行的值分为一组
df=DataFrame(np.random.randn(6,4),index=['one','two','three','four','five','six'],columns=pd.Index(['A','B','C','D'],name='Genus') )
df
df.plot(kind='bar')
df.plot(kind='barh')
#stacked=True即可为DataFrame
#生成堆积柱状图，这样每行的值就会被堆积在一起
df.plot(kind='barh',stacked=True,alpha=0.5)

#%%

#计算两个（或更多）因素的简单交叉表。
#crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=False, dropna=True, normalize=False)
#默认情况下，计算一个因子的频率表
filepath='C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch08PlottingAndVisualzation/data'
tips=pd.read_csv(filepath+'/tips.csv')
party_counts=pd.crosstab(tips['day'],tips['size'])
party_counts


#%%
#一个人和6个人的聚会比较少，因此去掉
party_counts=party_counts.ix[:,2:5]
#规格化，使得各行的和为1
#DataFrame.div(other, axis=’columns’, level=None, fill_value=None)
#等于dataframe/other
party_pcts=party_counts.div(party_counts.sum(1).astype(float),axis=0)

party_pcts
party_pcts.plot(kind='bar',stacked=True)

party_pcts.plot(kind='barh',stacked=True)


#直方图和密度图

#%%
# Serise.hist方法绘制直方图
tips['tip_pct']=tips['tip']/tips['total_bill']
tips['tip_pct'].hist(bins=50)
tips['tip_pct']

pd.crosstab(tips['sex'],tips['smoker'])

#%%
# 密度图
# 通过计算“可能会产生观测数据的连续概率分布的估计”
# 调用plot(kind='kde')即可生成一张密度图
tips['tip_pct'].plot(kind='kde')


# 将直方图和密度图画在一起
#%%
comp1=np.random.normal(0,1,size=200) #N(0,1)
comp2=np.random.normal(10,2,size=200) #N(10,4)

#  np.comcatenate([a1,a2,...],axis=0)
#  沿着给定轴对给定一组队列进行连接
values=Series(np.concatenate([comp1,comp2]))
values
values.hist(bins=100,alpha=0.3,color='k',normed=True)

values.plot(kind='kde',style='k--')


#散布图 scatter plot
#%%
macro=pd.read_csv(filepath+'/macrodata.csv')
data=macro[['cpi','m1','tbilrate','unemp']]
trans_data=np.log(data).diff().dropna()
trans_data[-5:]

plt.scatter(trans_data['m1'],trans_data['unemp'])
plt.title('Change in log %s vs. log %s' %('m1','unemp'))
plt.scatter(trans_data['m1'],trans_data['unemp'])

# diagonal 取不同值对角线上的图形是不同的
# 默认为hist 直方图，kde为密度图，形状与正态分布类似
pd.scatter_matrix(trans_data,diagonal='kde',c='k',alpha=0.3)

