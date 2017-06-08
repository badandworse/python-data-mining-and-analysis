import numpy as np
import pandas as pd
from pandas import DataFrame ,Series
import json
import matplotlib
filepath2='C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch07Dataprocessing/'

db=json.load(open(filepath2+'foods-2011-10-03.json'))
len(db)

db[0].keys()
db[0]['nutrients'][0]
nutrients=DataFrame(db[0]['nutrients'])
len(nutrients)
nutrients[:7]
info_keys=['description','group','id','manufacturer']
info=DataFrame(db,columns=info_keys)
info[:5]
info

pd.value_counts(info.group)[:10]

#%%
nutrients=[]
for rec in db:
    # 将数据中的nutrients字典提出来单独生成一个DataFrame
    # 并将'id'加入到DataFrame中
    fnuts=DataFrame(rec['nutrients'])
    fnuts['id']=rec['id']
    nutrients.append(fnuts)

# 使用concat将各个id的nutrients DataFrame合并起来
nutrients=pd.concat(nutrients,ignore_index=True)
len(nutrients)

# 发现有重复项
#%%
nutrients.duplicated().sum()
nutrients=nutrients.drop_duplicates()

col_mapping={'description':'food',
             'group':'fgroup'   }
info=info.rename(columns=col_mapping,copy=False)
info.columns
col_mapping={'description':'nutrient',
             'group':'nutgroup'   }

nutrients=nutrients.rename(columns=col_mapping,copy=False)
nutrients

#将info和nutrients合并
ndata=pd.merge(nutrients,info,on='id',how='outer')
ndata.index

ndata.ix[30000]

#根据营养和食物分组
result=ndata.groupby(['nutrient','fgroup'])['value'].quantile(0.5)
result
result['Zinc, Zn'].order().plot(kind='barh')

by_nutrient=ndata.groupby(['nutgroup','nutrient'])
get_maximum=lambda x:x.xs(x.value.idxmax())
get_minimum=lambda x:x.xs(x.value.idxmin())

max_foods=by_nutrient.apply(get_maximum)[['value','food']]

max_foods
max_foods.food=max_foods.food.str[:50]
max_foods.ix['Amino Acids']['food']
result1=ndata.groupby(['nutrient','fgroup'])
result1.quantile(0.5)
result1['fgroup']

df = DataFrame(np.array([[1, 1], [2, 10], [3, 100], [4, 100]]),
                   columns=['a', 'b'])
df
df.quantile(0.2)
df.quantile(0.25)
df.quantile(0.75)

df.quantile(0.5)

