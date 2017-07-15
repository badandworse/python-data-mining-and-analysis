#Date and Time Data Types and Tools
#%%
from datetime import datetime 
import pandas as pd 
import numpy as np
now =datetime.now()
now
now.year,now.month,now.day

now.year,now.day

# datetime以毫秒形式存储日期和时间
# datetime.timedelta表示
# 两个datetime对象之间的时间差
delta=datetime(2011,1,7)-datetime(2008,6,24,8,15)
delta
type(delta)

delta.days
delta.seconds

#%%
# datetime对象加上一个或多个timedelta
# 会产生一个新对象
from datetime import timedelta
start =datetime(2011,1,7)
start+timedelta(12)

#字符串和datetime的相互转换
#  利用str或strftime方法，
#  datetime对象和pandas的Timestamp对象
stamp=datetime(2011,1,3)
str(stamp)
stamp.strftime('%Y-%M-%D')
# datetime.strptime
# 用这些格式化编码将字符串转换为日期,
# 该函数需要给出需要转换的时间字符串的格式
value='2011-01-03'
datetime.strptime(value,'%Y-%m-%d')

datestrs=['7/6/2011','8/6/2011']

[datetime.strptime(x,'%m/%d/%Y') for x in datestrs]

#%%
# dateutil.parser.parse 
# 可以直接解析一些常见的格式
from dateutil.parser import parse
parse('2011-01-03')
parse('Jan 31,1997 10:45 PM')


# 传入dayfirst=True 即可解决日通常出现在月前面
parse('6/12/2011',dayfirst=True)

datestrs

# pandas 通常处理成组日期的，
# 不管这些日期是DateFrame的轴索引还是列
pd.to_datetime(datestrs)

idx=pd.to_datetime(datestrs+[None])
idx
pd.isnull(idx)

#%%
#时间序列基础
from datetime import datetime
from pandas import DataFrame,Series
dates=[datetime(2011,1,2),datetime(2011,1,5),datetime(2011,1,7),
       datetime(2011,1,8),datetime(2011,1,10),datetime(2011,1,12) ]

ts=Series(np.random.randn(6),index=dates)
ts

type(ts)
ts.index

# 不同索引的时间序列之间的算术运算会自动按日期对齐:
ts+ts[::2]
ts.index.dtype
stamp=ts.index[0]
stamp

#索引、选取、子集构造
stamp=ts.index[2]
ts[stamp]
# 对于带有DatetimeIndex 索引的Series
# 只要传入一个可以被解释为日期的的字符串
# 就可以读取值
ts['1/10/2011']
ts['20110110']
#%%
longer_ts=Series(np.random.randn(1000),
                 index=pd.date_range('1/1/2000',periods=1000))

longer_ts

# 对于较长的时间序列，只需传入“年”或“年月”即可轻松选取数据的切片
longer_ts['2001']
ts[datetime(2011,1,7)]
# 用不存在于该时间序列中的时间戳对其进行切片
ts['1/6/2011':'1/11/2011']

ts.truncate(after='1/9/2011')

#%%
dates=pd.date_range('1/1/2000',periods=100,freq='W-WED')

long_df=DataFrame(np.random.randn(100,4),
                  index=dates,
                  columns=['Colorado','Texas','New York','Ohio']  )

long_df.ix['5-2001']
dates[-5:]


#带有重复索引的时间序列
dates=pd.DatetimeIndex(['1/1/2000','1/2/2000','1/2/2000','1/2/2000','1/3/2000'])
dup_ts=Series(np.arange(5),index=dates)
dup_ts.index.is_unique
# 对这个时间序列进行索引，要么产生标量值，要么产生切片
dup_ts['1/3/2000']
dup_ts['1/2/2000']
grouped=dup_ts.groupby(level=0)
grouped.mean()
grouped.size()
