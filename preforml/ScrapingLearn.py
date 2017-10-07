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



#chapter5
#存储数据
##媒体文件
# urlib.request.urlretrieve可以根据文件的url下载文件
#%%
from urllib.request import urlretrieve

html=urlopen('http://www.pythonscraping.com/')
bsObj=BeautifulSoup(html)
imagesLocation=bsObj.find('a',{'id':'logo'}).find('img')['src']
urlretrieve(imagesLocation,'C:/Users/xg302/git/python-data-mining-and-analysis/preforml/logo.jpg')

urlretrieve('http://seanlahman.com/files/database/lahman-csv_2014-02-14.zip','data.zip')


#把数据存储到CSV

#%%
import csv
import os
#print(os.getcwd())

csvFile=open('c:/Users/xg302/git/python-data-mining-and-analysis/preforml/data/test.csv','w+')
try:
    writer=csv.writer(csvFile)
    writer.writerow(('number','number plus 2','number times 2'))
    for i in range(10):
        writer.writerow((i,i+2,i*2))

finally:
    csvFile.close()


#%%
html=urlopen('https://en.wikipedia.org/wiki/Comparison_of_text_editors')
bsObj=BeautifulSoup(html,'lxml')
table=bsObj.findAll('table',{'class':'wikitable'})[0]
rows=table.findAll('tr')
table
csvFile=open('c:/Users/xg302/git/python-data-mining-and-analysis/preforml/data/editors.csv','wt',newline='',encoding='utf-8')
writer=csv.writer(csvFile)
try:
    for row in rows:
        csvRow=[]
        for cell in row.findAll(['td','th']):
            csvRow.append(cell.get_text())
        writer.writerow(csvRow)
finally:
    csvFile.close()


#5.3 mysql

import pymysql

# 连接数据库
conn=pymysql.connect(user='root',password='654321',database='scraping')
#创建光标
cur=conn.cursor()
#%%
cur.execute("SELECT * FROM pages WHERE id=3")
print(cur.fetchall())
cur.close()
conn=cur.close()


#mail
#%%
import smtplib
from email.mime.text import MIMEText
from email.utils import parseaddr,formataddr
from email.header import Header
import time

#格式化一个邮件地址，因为如果包含中文，需要通过Header对象进行编码
def _format_addr(s):
    name,addr=parseaddr(s)
    return formataddr((Header(name,'utf-8').encode(),addr))

from_addr='675714883@qq.com'
password='eahulorcalbfbeia'
to_addr='1311469822@qq.com'
smtp_server='smtp.qq.com'
msg=MIMEText("The body of the email is here")
msg['Subject']="An Email Alert"
msg['From']=_format_addr('Python爱好者<%s>'%from_addr)
msg['To']=_format_addr('管理员<%s>' %to_addr)
msg['Subject']=Header('来自SMTP的问候...','utf-8').encode()

server=smtplib.SMTP_SSL(smtp_server,465) 
server.set_debuglevel(1)
server.login(from_addr,password)
server.sendmail(from_addr,[to_addr],msg.as_string())
server.quit()

#%%
#价格提醒
def sendmail(subject,body):
    msg=MIMEText(body)
    msg['Subject']=subject
    msg['From']='675714883@qq.com'
    msg['To']='1311469822@qq.com'

    server=smtplib.SMTP_SSL(smtp_server,465)
    server.set_debuglevel(1)
    server.login(from_addr,password)
    server.sendmail(from_addr,[to_addr],msg.as_string())
#%%
html=urlopen('https://www.sonkwo.com/products/2267?game_id=2267')
bsObj=BeautifulSoup(html,'lxml')
price_s=bsObj.find('span',{'class':'sale-price'}).get_text()
price=int(price_s[1:3])
while(price>129):
    bsObj=BeautifulSoup(html,'lxml')
    price_s=bsObj.find('span',{'class':'sale-price'}).get_text()
    price=int(price_s[1:3])
    time.sleep(3600)

sendmail("dishonored good price",'just buy it ')

#chapter6 读取文档
#%%
ll=[1,2,3,4]
sum([i for i in range(100)])
