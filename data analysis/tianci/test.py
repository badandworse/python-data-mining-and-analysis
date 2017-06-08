#%%
import pandas as pd
import numpy as np
from pandas import DataFrame,Index

# 首先读出数据，然后将date列单出取出来，利用正则表达式将9月份的日期取出来
# 由于正则表达式得到的是list，所以必须把series date的各选项取到list里，再利用是否为空得到符合条件的index
# 最后利用DateFrame.ix取出所需的数据
# 待做
#%%
filepath2='C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/tianci/'
df=pd.read_csv(filepath2+'Tianchi_power.csv')
len(df)
df.columns
df.head()
df.groupby('user_id').mean().sum()
type(df['record_date'][0])

import re
pattern=r'[0-9]{4}/9/[0-9]{1,2}'
regex=re.compile(pattern)
# findall会返回元素列表

#%%
m=df['record_date']
m=m.str.findall(pattern)
index_l=[]
for l in m:
    index_l.append(''.join(l))
print('over')
index_l[256]
index_l.index(index_l[256])
#%%
index_list=[]
q=len(index_l)-1
while q>=0:
    if index_l[q]!='':
        index_list.append(q)
    q=q-1
print('over')
#%%
len(index_list)
pdf1=df.ix[index_list]
pdf1.to_csv(filepath2+'only_9.csv')
#%%
print(0)

        
df['record_date'][256]
df.ix[[255,256,257]]