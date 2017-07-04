#透视表
#%%
import numpy as np 
import pandas as pd
from pandas import DataFrame,Series

#%%
filepath='C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch09Data Aggregation and Group Operations/data'
tips=pd.read_csv(filepath+'/tips.csv')
tips['tip_pct']=tips['tip']/tips['total_bill']


tips[:5]
# 透视表默认的聚合类型是分组平均数
#  代码是以'sex'和'smoker'聚合，如果为index=['sex','smoker ']
#  如果为index=['sex','smoker ']，聚合结果将sex和smoker放在行，
#  求其他列在在各分类结果的平均值
tips.pivot_table(index=['sex','smoker'])
#  如果为columns=['smoker'],则将smoker放在列，再聚合
tips.pivot_table(index=['sex'],columns=['smoker'])

#%%
# 选取部分列来聚合,将需要聚合的列以列表形式赋给values
tips.pivot_table(values=['tip_pct','size'],index=['sex','day'],
                 columns=['smoker'])


#%%
# margin=True时，添加分享小计。这将会添加标签为All的行和列
# 其值对应于单个等级中所有数据的统计，不单独考虑聚合项
#  代码中All列没有考虑烟民与非烟民，而All行则没考虑性别和天
tips.pivot_table(values=['tip_pct','size'],index=['sex','day'],
                 columns=['smoker'],margins=True)


# 其他聚合函数，将其传给aggfunc即可
#  代码传入len(count亦可)得到有关分组大小的交叉表
tips.pivot_table(values='tip_pct',index=['sex','smoker'],columns='day',aggfunc=len,margins=True)

# 存在无效值，使用fill_value为其填充值
tips.pivot_table(values='size',index=['time','sex','smoker'],columns='day',aggfunc='sum',fill_value=0)


#pivot_table参数: values:要聚合的列名，默认为全选；
# index:分组列，并转为行;
# columns:分组列，最终表中为列;
# aggfunc：能用作grouped的函数，默认为numpy.mean;
# fill_value:填充无效值；
# margins:添加行/列小计和总计，默认为False;
# dropna:是否包含含有缺失值的列，默认为True；
# margin_name:需要统计的行或列的名字
