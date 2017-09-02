#%%
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import pandas as pd
from pandas import Series,DataFrame

#%%
html =urlopen("http://www.pythonscraping.com/pages/warandpeace.html")
bsObj=BeautifulSoup(html,"lxml")
nameList=bsObj.find_all("span",{"class":"green"})
#%%
#这里是要取出得到标签的文本内容
#故使用get_text()
for name in nameList:
    print(name.get_text())

nameList[0]

bsObj.find_all(text="the prince")

allText=bsObj.find_all(id='text')
allText[0].get_text()

#%%
#处理兄弟标签 next_sibling()函数
# 第一行的表格不会被打印，
# 因为对象不能把自己作为兄弟标签
html=urlopen('http://www.pythonscraping.com/pages/page3.html')
bsObj=BeautifulSoup(html,'lxml')
for sibling in bsObj.find('table',{'id':'giftList'}).tr.next_siblings:
    print(sibling)


#父标签处理
#parent and parents functions

#正则表达式和BeautifulSoup
images=bsObj.find_all("img",{"src":re.compile("\.\.\/img\/gifts\/img.*\.jpg")})
for img in images:
    print(img['src'])

#获取属性
# 对于一个标签对象，使用以下代码可以获取它的全部对象
# mytag
images[0].attrs
bsObj.find('table',{'id':'giftList'}).attrs

#Lambda表达式
#findAll允许把特定函数类型当作findAll函数的参数
# 唯一限制是必须把一个标签作为参数且返回结果是布尔类型
# soup.findAll(lambda tag:len(tag.attrs)==2)
# 找出有两个属性的标签



#Chapter3开始采集
#%%
#提取页面内所有链接
html=urlopen('https://en.wikipedia.org/wiki/Kevin_Bacon')
bsObj=BeautifulSoup(html,'lxml')
for link in bsObj.find('div',{'id':'bodyContent'}).find_all('a',href=re.compile('^(/wiki/)((?!:).)*$')) :
    if 'href' in link.attrs:
        print(link.attrs['href'])


m="12345"
m.replace("http://","")

import scrapy


#豆瓣250
#%%
global movieList
def getMoveList(stringPath,nameList,rateList):
    m=re.compile('^(\\xa0)')
    url="https://movie.douban.com/top250"+stringPath;
    html=urlopen(url)
    bsObj=BeautifulSoup(html,'lxml')
    for movie in bsObj.find_all('span',{'class':'title'}):
        if m.match(movie.get_text()) is None:
            nameList.append(movie.get_text())
    for rate in bsObj.find_all('span',{'class':'rating_num'}):
        rateList.append(rate.get_text())
    mm=bsObj.find('span',{'class':'next'})
    return mm.a

#%%
nameList=[]
rateList=[]
link=getMoveList('?start=25&filter=',nameList,rateList)

#%%
while True:
    link=getMoveList(link['href'],nameList,rateList)
    i=i+1
    print(i)
    if link is None:
        break



index_NO=[i for i in range(1,251)]
movieDT=pd.DataFrame({'No':pd.Series(index_NO),
                      'MovieName':pd.Series(nameList),
                      'Ave_Rate':pd.Series(rateList)                   
                     })
movieDT

#完成，不过被ban了，暂时无法测试

#chapter4
# 使用twitter api
#%%
from twitter import Twitter,OAuth

t=Twitter(auth=OAuth('875557848-Yum5KsTWxMU4QcaNxrVaRpr0uExfxsqBwezTJtfw','Ka3wEQcnOOMUfjMedzW5eQJUwXrvnOqQSnn4ojEX97Fmo',
                     	'mRFT5WJNft8t63EWM5fWhgPNj','5uPR9gtFFtYpjgvfkZf5AS5XhQT20jAi0ecgYG25GqR8fD9zpl'
                    ))
pythonTweets=t.search.tweets(q="#python")
# 更新时间线,发推
statusUpdate=t.statuses.update(status='hello,world')
print(pythonTweets)
# 获取指定用户的一组推文
pythonStatus=t.statuses.user_timeline(screen_name="montypython",count=5)
print(pythonStatus)

# 使用新浪api
#  todo