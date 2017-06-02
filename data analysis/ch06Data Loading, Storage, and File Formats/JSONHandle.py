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
