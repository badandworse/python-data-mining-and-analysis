#%%

#JSON
# json.loads可将JSON转换成Python形式
obj="""
{
    "names":"Wes",
    "places_lived":["United States","Spain","Germany"],
    "pet":null,
    "siblings":[{"name":"Scott","age":25,"pet":"Zuko"},
                {"name":"Katie","age":33,"pet":"Cisco"}   ]
}
"""
import json
import pandas as pd
from pandas import DataFrame ,Series
result=json.loads(obj)
type(result)

# json.dumps则将python对象转换成JSON格式
asjosn=json.dumps(result)
asjosn

# 如何将JSON对象转换为DataFrame：
# 向DataFrame构造器传入一组JSON对象，并选取数据字段的子集
siblings=DataFrame(result['siblings'],columns=['name','age'])
siblings

#%%
#XML和HTML：web信息收集
from lxml.html import parse
from urllib.request import  urlopen

parsed=parse(urlopen('http://finance.yahoo.com/q/op?s=AAPL+Options'))
doc=parsed.getroot()
links=doc.findall('.//a')
links[15:20]
'''
运行结果:
[<Element a at 0x22f9b0927c8>,
 <Element a at 0x22f9b092818>,
 <Element a at 0x22f9b092868>,
 <Element a at 0x22f9b0928b8>,
 <Element a at 0x22f9b092908>]

'''

lnk=links[28]
lnk


#%%
#使用HTML和Web API
import requests

url='http://search.twitter.com/search.json?q=python%20pandas'
resp=requests.get(url)

resp

#%%
#使用数据库
import sqlite3

query='''CREATE TABLE test(a VARCHAR(20),b VARCHAR(20),c REAL, d INTEGER);'''
con=sqlite3.connect(':memory:')
con.execute(query)
con.commit()

data=[('Atlanta','Georgia',1.25,6),
        ('Tallahassee','Florida',2.6,3),
        ('Sacramento','California',1.7,5)]
stmt="INSERT INTO test VALUES(?,?,?,?)"
con.executemany(stmt,data)
con.commit()

cursor=con.execute('select * from test')
rows=cursor.fetchall()
rows

#获取列名
cursor.description

DataFrame(rows,columns=list(zip(*cursor.description))[0])
list(zip(*cursor.description))

#pandas.io.sql 中的read_sql_query 只需传入select语句和连接对象即可
import pandas.io.sql as sql
sql.read_sql_query('select * from test',con)