import pandas as pd
import numpy as np
from matplotlib import pyplot as pl
import json


names1880=pd.read_csv('C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch02/data/names/yob1880.txt',names=['name','sex','births'])

names1880

names1880.groupby('sex').births.sum()
years=range(1880,2011)
pieces=[]
columns=['name','sex','birth']
for year in years:
    path='C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch02/data/names/yob%d.txt' %year
    frame=pd.read_csv(path,names=columns)

    frame['year']=year
    pieces.append(frame)

#%%
#将多个DataFrame组合到一起，并不保留read_csv所返回的原始行号
names=pd.concat(pieces,ignore_index=True)
names

#%%

total_births=names.pivot_table('birth',index='year',columns='sex',aggfunc='sum')
total_births.tail()

total_births.plot(title='Total birth by sex and year')

def add_prop(group):
    #整数除法会向下圆整
    births=group.birth.astype(float)
    group['prop']=births/births.sum()
    return group

names=names.groupby(['year','sex']).apply(add_prop)
names

#用np.allclose来检查这个分组总计值是否足够接近1
np.allclose(names.groupby(['year','sex']).prop.sum(),1)


#%%
def get_top1000(group):
    return group.sort_index(by='birth',ascending=False)[:1000]

grouped=names.groupby(['year','sex'])
top1000=grouped.apply(get_top1000)
top1000

#%%
pieces=[]
for year,group in names.groupby(['year','sex']):
    pieces.append(group.sort_index(by='birth',ascending=False)[:1000])

top1000=pd.concat(pieces,ignore_index=True)
top1000

#%%
boys=top1000[top1000.sex=='M']
girl=top1000[top1000.sex=="F"]
total_births=top1000.pivot_table('birth',index='year',columns='name',aggfunc='sum')
total_births

#%%
subset=total_births[['John','Harry','Mary','Marilyn']]
subset.plot(subplots=True,figsize=(12,10),grid=False,title='Number of births per year')