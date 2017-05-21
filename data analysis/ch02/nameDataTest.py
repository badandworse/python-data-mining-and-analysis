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


#%%
#评估命名多样性的增长
table=top1000.pivot_table('prop',index='year',columns='sex',aggfunc='sum')
table.plot(title='Sum of table1000.prop by year and sex',yticks=np.linspace(0,1.2,13),xticks=range(1880,2020,10))

df=boys[boys.year==2010]
df

prop_cumsum=df.sort_index(by='prop',ascending=False).prop.cumsum()
prop_cumsum[:20]
prop_cumsum.searchsorted(0.5)

df=boys[boys.year==1900]
in1900=df.sort_index(by='prop',ascending=False).prop.cumsum()
in1900.searchsorted(0.5)+1

def get_quantitle_count(group,q=0.5):
    group=group.sort_index(by='prop',ascending=False)
    return group.prop.cumsum().searchsorted(q)+1


#得到的数据想是string，因此需要使用astype将其数据转换为 numeric data
diversity=top1000.groupby(['year','sex']).apply(get_quantitle_count)
diversity=diversity.unstack('sex')
diversity.head()
diversity=diversity.astype(float)
diversity.plot(title='number of popular names in top 50%')


#"最后一个字母"的变革

#%%
#从name列中取出最后一个字母
get_last_letter=lambda x: x[-1]
last_letters=names.name.map(get_last_letter)
last_letters.name='last_letter'
table=names.pivot_table('birth',index=last_letters,columns=['sex','year'],aggfunc='sum')

subtable=table.reindex(columns=[1910,1960,2010],level='year')
subtable.head()
#%%
subtable.sum()
letter_prop=subtable/subtable.sum().astype(float)

fig,axes=pl.subplots(2,1,figsize=(10,8))
letter_prop['M'].plot(kind='bar',rot=0,ax=axes[0],title='Male')
letter_prop['F'].plot(kind='bar',rot=0,ax=axes[1],title='FeMale',legend=False)

letter_prop=table/table.sum().astype(float)
dny_ts=letter_prop.ix[['d','n','y'],'M'].T
dny_ts.head()
dny_ts.plot()

all_names=top1000.name.unique()
mask=np.array(['lesl' in x.lower() for x in all_names])

lesley_like=all_names[mask]
lesley_like

filtered=top1000[top1000.name.isin(lesley_like)]
filtered.groupby('name').birth.sum()

#%%
table=filtered.pivot_table('birth',index='year',columns='sex',aggfunc='sum')
table=table.div(table.sum(1),axis=0)
table.tail()

table.plot(style={'M':'k-','F':'k--'})

