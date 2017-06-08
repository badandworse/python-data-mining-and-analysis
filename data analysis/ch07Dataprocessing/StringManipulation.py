import numpy as np
from pandas import DataFrame,Series
import pandas as pd

#%%
#字符串对象方法
val='a,b, guido'
val.split(',')

# strip 去掉字符串中开头和结尾的空白符
# lstrip 去掉左边的  rstrip 去掉右边的
piece=[x.strip() for x in val.split(',')]
piece

first,second,third=piece
first+'::'+second+'::'+third

# 向字符串的join传入列表或元祖，可以在各个元素直接加入这个字符串形成一个新的字符串
#  将序列中的元素以指定的字符链接生成一个新的字符串
'::'.join(piece)


# python 检测子串的最佳方式是in 关键字
# index和find也行. 如果没有传入的字符串，index会报错,find会返回-1
# rfind 返回最后一个发现的字符的第一个字符所在的位置
'guido' in val
val.index(',')
val.find(':')
val.rfind(',')
# count计算子串出现的次数
val.count('p')

val.replace('a','ds',5)

'   dsds   dsd  dsds   '.strip()

#%%
#正则表达式
import re
# \t是制表符
text='foo bar \t baz \tqux'
text
re.split('\s+',text)

# 利用re.compile可以先编译好一个可重用的regex对象
regex=re.compile('\s+')
regex.split(text)

# findall得到匹配的所有模式
regex.findall(text)

text="""Dave dave@google.com
        Steve steve@gmail.com
        Rob rob@gmail.com
        Ryan ryan@yahoo.com
      """

pattern=r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
# re.IGNORECASE使得正则表达式对大小写不敏感
regex=re.compile(pattern,flags=re.IGNORECASE)
regex.findall(text)

# serach 只返回第一个匹配项
m=regex.search(text)
m
text[m.start():m.end()]

print(regex.match(text))

# sub，匹配到的内容转换为指定字符串
print (regex.sub('REDACTED',text))

# groups方法返回一个由模式各段组成的元祖
#  需要的分组，在模式中用括号括起来
pattern=r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
print(pattern)
regex=re.compile(pattern,flags=re.IGNORECASE)

m=regex.match('webm@bright.net')
m.groups()
# findall会返回一个元祖列表
regex.findall(text)
# sub 通过诸如\1,\2之类的特殊符号访问各匹配项中的分组

print (regex.sub(r'Username: \1,Domain:\2,Suffix:\3',text))

# ?P<name>regex 将匹配到的text加入到‘name’ 组中
regex=re.compile(r"""(?P<username>[A-Z0-9._%+-]+)@(?P<domain>[A-Z0-9.-]+\.(?P<suffix>[A-Z{2,4}]))""",flags=re.IGNORECASE|re.VERBOSE)
m=regex.match('webm@bright.net')
m.groupdict()

#%%
#pandas中矢量化的字符串函数
data={'Dave':'dave@google.com','Steve':'steve@gmail.com','Rob':'rob@gmail.com','Wes':np.nan}

data=Series(data)
data
# 通过data.map 所有字符串和正则表达式方法能悲惨应用与（传入lambda报大师或其他函数）各个值
# 但是如果是NA值就会报错。
# Series有一些能够跳过NA值的字符串操作方法。
# str.contains检查各个电子邮件是否含有'gamil'
data.isnull()
data.str.contains('gmail')

#  str.findall可以传入正则表达式应用与series的元素 ，还可以加入re选项(IGNRECASE):
pattern
data.str.findall(pattern,flags=re.IGNORECASE)

#  有两个办法可以实现矢量化的元素获取操作:
#    str.get 0得到第一列，1得到第二列，依次类推
#    在str属性上使用索引 .str[0] 同上
matches=data.str.match(pattern,flags=re.IGNORECASE)
matches

matches.str.get(0)
matches.str[0]
#   同样可以使用切片
matches.str[:5]

