import re
#%%
str='an example word:cat!!'
#搜需给定字符串中是否包含一下格式
# 'word:'+3个字母构成的字符串
#'r'开头标明保留后面给定的字符串，不要有任何改变
#此处是保留'\'
match=re.search(r'word:\w\w\w',str) 

if match:
    #如果匹配成功，re.search会将结果储存在match中
    #match.group() 取出匹配成功的字符串
    #如果匹配不成功，则为空
    print('found',match.group())  ##'found word:cat'
else:
    print('did not find')


## .=any char but \n
match=re.search(r'..g','piiig') #found,match.group()='iig'
match.group()

## \w{5} 匹配五个字符

#findall 找出给定字符串中所有符合条件的part，并组成一个list
## Suppose we have a text with many email addresses
str = 'purple alice@google.com, blah monkey bob@abc.com blah dishwasher'

emails=re.findall(r'[\w\.-]+@[\w\.-]+',str)
emails

#findall 如果将正则表达式一部分()起来，
#则不会返回一个由string组成的list
#而是返回一个由与()数量相同的元祖list组成
#每个tuple对应的是()那部分正则表达式匹配到的元祖
tuples=re.findall(r'([\w\.-]+)@([\w\.-]+)',str)
tuples

#贪婪匹配:正则匹配默认是贪婪匹配，即尽可能匹配多的字符串
#而要使用非贪婪匹配，则在那部分字符串后面加上'?'
##由于默认贪婪匹配，则\d+将字符串匹配完，0*则没匹配为空
##加上'?'后为非贪婪模式
re.match(r'^(\d+)(0*)$','1023000').groups()
re.match(r'^(\d+?)(0*)$','1023000').groups()

#re.sub()
#re.sub(pat,replacement,str)
##thr replacement string can include '\1','\2'
##which refer to the text from group(1) group(2)
print(re.sub(r'([\w\.]+)@([\w\.-]+)',r'\1@qq.com',str))