import pandas as pd
import numpy as np
import matplotlib
from  pandas import DataFrame,Series
import json

path='C:/Users/xg302/Documents/mystuff/python-data/pydata-book-master/ch02/usagov_bitly_data2012-03-16-1331923249.txt'
records=[json.load(line) for line in open(path)]
time_zone=[rec['tz'] for rec in records if 'tz' in rec]
frame=DataFrame(records)
#fillna函数填补缺省值
clean_tz=frame['tz'].fillna('Missing')
#利用bool表达式选出list中为空的值替换之
clean_tz[clean_tz=='']='Unknown'
tz_counts=clean_tz.value_counts()
tz_counts[:10].plot(kind='barh',rot=0)
