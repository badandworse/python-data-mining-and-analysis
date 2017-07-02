#%%
import pandas as pd 
import numpy as np 
from pandas import DataFrame,Series

#%%
filepath='C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch08PlottingAndVisualzation/data'
data =pd.read_csv(filepath+'/Haiti.csv')
data

data[['INCIDENT DATE','LATITUDE','LONGITUDE']][:10]
data['CATEGORY'][:6]
data[:10]
# 通过data.describe发现数据中的异常数据
#%%
data.describe()

# 根据describe的结果，对数据进行筛选
data=data[(data['LATITUDE']>18)&(data['LATITUDE']<20)&
            (data['LONGITUDE']>-75)&(data['LONGITUDE']<-70)&data['CATEGORY'].notnull()]


#%%
def to_cat_list(catstr):
    stripped=(x.strip() for x in catstr.split(','))
    return [x for x in stripped if x]

def get_all_categories(cat_series):
    cat_sets=(set(to_cat_list(x)) for x in cat_series)
    return sorted(set.union(*cat_sets))

def get_english(cat):
    code,names=cat.split('.')
    if '|' in names:
        names=names.split('|')[1]
    return code,names.strip()


get_english('2.Urgences logistiques | Vital Lines')

all_cats=get_all_categories(data['CATEGORY'])
all_cats[1]
english_mapping =dict(get_english(x) for x in all_cats)
english_mapping['2a']

english_mapping['6c']

english_mapping['4']

english_mapping

#%%
def get_code(seq):
    return [x.split('.')[0] for x in seq if x]

all_codes=get_code(all_cats)
len(all_codes)

code_index=pd.Index(np.unique(all_codes))
code_index
dummy_frame=DataFrame(np.zeros((len(data),len(code_index))),index=data.index,columns=code_index)

dummy_frame.ix[:,:6].info()
dummy_frame.T.ix[:,:6]

#%%

#把每一行存在的索引出设为1
for row ,cat in zip(data.index,data.CATEGORY):
    codes=get_code(to_cat_list(cat))
    dummy_frame.ix[row,codes]=1

data=data.join(dummy_frame.add_prefix('category_'))
data.info()

#%%
#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import   Basemap