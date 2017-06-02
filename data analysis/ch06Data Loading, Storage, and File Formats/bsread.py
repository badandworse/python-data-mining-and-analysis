import pandas as pd
from pandas import DataFrame,Series

#%%
filepath2='C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch06Data Loading, Storage, and File Formats/'
testkey=pd.read_csv(filepath2+'bs key.txt',header=None)
testkey
df=testkey.drop_duplicates()
df.to_csv(filepath2+'finished.csv')
