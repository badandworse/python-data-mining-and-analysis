#日期的范围、频率以及移动
import pandas as pd
import numpy as np
from pandas import DataFrame,Series
from datetime import datetime

#%%
dates=[datetime(2011,1,2),datetime(2011,1,5),datetime(2011,1,7),
       datetime(2011,1,8),datetime(2011,1,10),datetime(2011,1,12) ]

ts=Series(np.random.randn(6),index=dates)
ts

# resample 将时间序列转换成固定频率
# 'D' 就是每天
ts.resample('D').asfreq()

# 生成日期范围


#  date_range 可用于生产指定长度的 DatetimeIndex
#  传入开始和结束则会默认按照天的间隔产生,可更改freq参数值，默认为'D'
#  如果指传入起始或结束日期，那就还得传入一个表示一段时间的数字
index=pd.date_range('4/1/2012','6/1/2012')
index
pd.date_range(start='4/1/2012',periods=20)
pd.date_range(end='4/1/2012',periods=20)

#  生成一个由每月最后一个工作日组成的日期索引，
#  传入"BM"
#  freq='4h' 即为间隔为4h创建 dateindex，传入指定频率字符串
pd.date_range('1/1/2000','12/1/2000',freq='BM')
pd.date_range('1/1/2000','12/1/2000',freq='4h')

#  date_range 默认会保留起始和结束时间戳的时间信息
#  传入normalize=True，则会规范成午夜的时间戳
pd.date_range('5/2/2012 12:56:31',periods=5)
pd.date_range('5/2/2012 12:56:31',periods=5,normalize=True)


