#%%
import pandas as pd
import numpy as np
from pandas import DataFrame ,Series
import sys


fillpath='C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch06Data Loading, Storage, and File Formats/ex1.csv'
#read_csv将其读入一个DataFrame,自动按逗号分割
df=pd.read_csv('C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch06Data Loading, Storage, and File Formats/ex1.csv')
df
filepath2='C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch06Data Loading, Storage, and File Formats/'

testKey=pd.read_csv(filepath2+'bs.txt')
testKey

#read_table也可以，不过得指定分隔符，此处是‘,’
pd.read_table(fillpath,sep=',')

#%%
filepath2='C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch06Data Loading, Storage, and File Formats/'
#header=None时pandas为其分配默认的列名
pd.read_csv(filepath2+'ex2.csv',header=None)

pd.read_csv(filepath2+'ex2.csv',names=['a','b','c','d','message'])


#如果需要某列做索引，可以使用index_col来指定某列做索引,例子中是将message列做索引
pd.read_csv(filepath2+'ex2.csv',names=['a','b','c','d','message'],index_col='message')


#多个列做一个层次化索引，只需传入由列编号或列名组成的列表即可
parsed=pd.read_csv(filepath2+'csv_mindex.csv',index_col=['key1','key2'])
parsed
list(open(filepath2+'ex3.txt'))

#ex3.txt的间隔是由数量不定的空白符分隔，使用正则表达式\s+表示即可
#因为列名数量比数据列少，所以数据的第一列应该是DataFRAME的索引
result=pd.read_table(filepath2+'ex3.txt',sep='\s+')
result

#skiprows 赋值列表，可以帮助你跳过文件的某些行
pd.read_csv(filepath2+'ex4.csv',skiprows=[0,2,3])


#缺失值处理，na_values可以接受一组用来表示缺失值的字符串
result=pd.read_csv(filepath2+'ex5.csv')
result.isnull()
result=pd.read_csv(filepath2+'ex5.csv',na_values=['NULL'])
result

#表示在message这一列，'foo'和'NA'都是用来表示缺失值，
# 而'something'这一列则是'two'
sentinels={'message':['foo','NA'],'something':['two']}
pd.read_csv(filepath2+'ex5.csv',na_values=sentinels)


#%%
#逐块读文本文件
result=pd.read_csv(filepath2+'ex6.csv')
result
# 如果只是想读取几行（避免读取整个文件），通过nrows进行指定即可
pd.read_csv(filepath2+'ex6.csv',nrows=5)

chunker=pd.read_csv(filepath2+'ex6.csv',chunksize=1000)
chunker

tot=Series([])
for piece in chunker:
    tot=tot.add(piece['key'].value_counts(),fill_value=0)
tot=tot.order(ascending=False)

tot[:10]
type(tot)
Series.count(tot)
tot.cumsum()

#%%
# 将数据写出到文本格式
data=pd.read_csv(filepath2+'ex5.csv')
data
data.to_csv(filepath2+'out.csv')

#  sys.stdout代表仅打印文本结果
data.to_csv(sys.stdout,sep='|')

#  缺失值在输出结果中默认为' ',通过给na_rep赋值而指定缺失值表示
data.to_csv(sys.stdout,na_rep='NULL')

data.to_csv(sys.stdout,index=False,header=False)

#  可以只写出一部分列，并指定排列顺序
data.to_csv(sys.stdout,index=False,columns=['a','b','c'])\

#  Series也有一个to_csv
dates=pd.date_range('1/1/2000',periods=7)
dates
ts=Series(np.arange(7),index=dates)
ts.to_csv(filepath2+'tseries.csv')

Series.from_csv(filepath2+'tseries.csv',parse_dates=True)

#%%
# 手动处理分隔符格式

#  对于任意单字符分割符文件，可以直接使用Python内置的csv模块
#  reader进行迭代将会每行产生一个列表
import csv
f=open(filepath2+'ex7.csv')
reader=csv.reader(f)
for line in reader:
    print (line , type(line))

lines=list(csv.reader(open(filepath2+'ex7.csv')))
header,values=lines[0],lines[1:]
data_dict={h:v for h,v in zip(header,zip(*values))}
data_dict
