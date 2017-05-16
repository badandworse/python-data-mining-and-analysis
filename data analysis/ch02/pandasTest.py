import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  pandas import DataFrame,Series
import json
#%%
print('hello world')
#<codecell>
path='C:/Users/xg302/Documents/mystuff/python-data/pydata-book-master/ch02/usagov_bitly_data2012-03-16-1331923249.txt'
records=[json.loads(line) for line in open(path)]
frame=DataFrame(records)
#fillna函数填补缺省值
clean_tz=frame['tz'].fillna('Missing')
#利用bool表达式选出list中为空的值替换之
clean_tz[clean_tz=='']='Unknown'
tz_counts=clean_tz.value_counts()
tz_counts[:10].plot(kind='barh',rot=0)

frame['a'][51]
#舍弃dataframe中没有a属性值的元素
results=Series(x.split()[0] for x in frame.a.dropna())
results[:5]
#根据统计次数排序，并取前8
results.value_counts()[:8]

cframe=frame[frame.a.notnull()]

#%%
#如果包含windows则返回windows，否则返回not windows
operating_system=np.where(cframe['a'].str.contains('Windows'),'Windows','Not windows')
operating_system[:5]

#%%
#根据时区和新得到的操作系统列表对数据进行分组:
by_tz_os=cframe.groupby(['tz',operating_system])
agg_counts=by_tz_os.size().unstack().fillna(0)
agg_counts[:10]
indexer=agg_counts.sum(1).argsort()
indexer[:10]
count_subset=agg_counts.take(indexer)[-10:]
count_subset

count_subset.plot(kind='barh',stacked=True)
#%%
#按比例分组后画图根据清晰，windows+not windows的sum为1
normed_subset=count_subset.div(count_subset.sum(1),axis=0)
normed_subset.plot(kind='barh',stacked=True)