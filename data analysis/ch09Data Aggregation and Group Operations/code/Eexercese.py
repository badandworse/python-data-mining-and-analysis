# 2012联邦选举委员会数据库
# 利用map与dict映射，归纳处理数据
 
#%%
import numpy as np 
import pandas as pd
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
#%%
filepath='C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch09Data Aggregation and Group Operations/data'
fec=pd.read_csv(filepath+'/P00000001-ALL.csv')
fec.info()
fec.ix[123456]

unique_cands=fec.cand_nm.unique()
#unique_cands

#%%
parties={'Bachmann, Michelle':'Republican',
         'Cain, Herman':'Republican',
         'Gingrich, Newt':'Republican',
         'Huntsman, Jon':'Republican',
         'Johnson, Gary Earl':'Republican',
         'McCotter, Thaddeus G':'Republican',
         'Obama, Barack':'Democrat',
         'Paul, Ron':'Republican',
         'Pawlenty, Timothy':'Republican',
         'Perry, Rick':'Republican',
         "Roemer, Charles E. 'Buddy' III":'Republican',
         'Romney, Mitt':'Republican',
         'Santorum, Rick':'Republican'}

fec.cand_nm[123456:123461]


fec.cand_nm[123456:123461].map(parties)

fec['party']=fec.cand_nm.map(parties)
fec['party'].value_counts()
# 通过检查各分组中
# 非无效值的数量来查看分组元素有哪些错误
fec.groupby('cand_nm')['party'].count()

# 资助有退款，因此可能出现负的出资额
# 去除负的数据，只保留正的数据
fec=fec[fec['contb_receipt_amt']>0]
(fec['contb_receipt_amt']>0).value_counts()

# 主要候选人只有两位，因此提取出来
fec_mrbo=fec[fec.cand_nm.isin(['Obama, Barack','Romney, Mitt'])]
fec_mrbo.info()
fec.contbr_occupation.value_counts()[:10]

#%%
occ_mapping ={
    'INFORMATION REQUESTED PER BEST EFFORTS':'NOT PROVIDED',
    'INFORMATION REQUESTED' : 'NOT PROVIDED',
    'INFORMATION REQUESTED (BEST EFFORTS)':'NOT PROVIDED',
    'C.E.0.' :'CE0'
    }



# 如果没提供相关映射，则返回x
f=lambda x :occ_mapping.get(x,x)
fec['contbr_occupation']=fec['contbr_occupation'].map(f)
#%%
emp_mapping ={
    'NFORMATION REQUESTED PER BEST EFFORTS':'NOT PROVIDED',
    'INFORMATION REQUESTED':'NOT PROVIDED',
    'SELF':'SELF-EMPLOYED',
    'SELF EMPLOYED':'SELF-EMPLOYED'
    }


f=lambda x :emp_mapping.get(x,x)
fec.contbr_occupation=fec.contbr_occupation.map(f)

#%%
by_occupation=fec.pivot_table(values='contb_receipt_amt',index='contbr_occupation',
                              columns='party',aggfunc='sum')

# 过滤掉总出资额不足200w的数据
over_2mm=by_occupation[by_occupation.sum(1)>200000]
over_2mm

over_2mm.plot(kind='barh')
plt.show()
plt.show()
print(0)

#%%
def get_top_amounts(group,key,n=5):
    totals=group.groupby(key)['contb_receipt_amt'].sum()
    # 根据key对totals进行降序排序
    return totals.order(ascending=False)[n:]


grouped=fec_mrbo.groupby('cand_nm')
grouped.apply(get_top_amounts,'contbr_occupation',n=7)