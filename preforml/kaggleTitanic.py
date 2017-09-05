'''
问题概述:
分析哪种类型的乘客更有可能活下来
'''
#%%
import numpy as np
import pandas as pd
from pandas import DataFrame,Series

data=pd.read_csv('train.csv')
data.head()
data.info()
data.index
#观看各项属性的基本跟情况
data.describe()


#%%
#画出各数据与survived的关系
import matplotlib.pyplot as plt
import matplotlib


plt.rcParams['font.sans-serif']=['SimHei']   #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False     #用来正常显示正负号


fig=plt.figure()
fig.set(alpha=0.2) #设定图表颜色alpha颜色

#%%
#survived柱状图
plt.subplot2grid((3,5),(0,0))
data.Survived.value_counts().plot(kind='bar')
plt.title(u'获救情况(1为获救)') #标题
plt.ylabel(u'人数')



#Pclass等级柱状图
plt.subplot2grid((3,5),(0,2))
data.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"人数")
plt.title(u"乘客等级分布")


#Age and survived 图
plt.subplot2grid((3,5),(0,4))
plt.scatter(data.Survived,data.Age)
plt.ylabel(u"年龄")
plt.grid(b=True,which='major',axis='y')
plt.title(u"按年龄看获救分布(1为获救)")

#各等级的乘客年龄分布
plt.subplot2grid((3,5),(2,0),colspan=3)
data.Age[data.Pclass==1].plot(kind='kde')
data.Age[data.Pclass==2].plot(kind='kde')
data.Age[data.Pclass==3].plot(kind='kde')
plt.xlabel(u"年龄")
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
plt.legend((u"头等舱",u"2等舱",u"3等舱"),loc='best') #sets our legend for our graph


#各登船口岸上船人数
plt.subplot2grid((3,5),(2,4))
data.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上传人数")
plt.ylabel(u"人数")
plt.show()


#属性与获救结果的关联统计
#%%
fig=plt.figure()
fig.set(alpha=0.2) #设定图标颜色alpha

#舱位级别与获救与否的关系图

Survived_0=data.Pclass[data.Survived==0].value_counts()
Survived_1=data.Pclass[data.Survived==1].value_counts()
df=pd.DataFrame({u'获救':Survived_0,u'未获救':Survived_1})
df.plot(kind='bar',stacked=True)
plt.title(u'各个乘客等级的获救情况')
plt.xlabel(u'乘客等级')
plt.ylabel(u'人数')
plt.show()
df
data.info()

#性别与获救与否的关系图
#%%
Survived_m=data.Survived[data.Sex=='male'].value_counts()
Survived_f=data.Survived[data.Sex=='female'].value_counts()
df=pd.DataFrame({u'男性':Survived_m,u'女侠':Survived_f})
df.plot(kind='bar',stacked=True)
plt.title(u"按性别看获救情况")
plt.xlabel(u"性别")
plt.ylabel(u'人数')
plt.show()

#%%
#各种舱级别情况下个性别的获救情况
fig=plt.figure(figsize=(8,6))
fig.set(alpha=0.65)
plt.title(u'根据舱等级和性别的获奖情况')

ax1=fig.add_subplot(141)
data.Survived[data.Sex=='female'][data.Pclass!=3].value_counts().plot(kind='bar',label='female highclass',color='#FA2479')
ax1.set_xticklabels([u"获救",u"未获救"],rotation=0)
ax1.legend([u"女性/高级舱"],loc='best')

ax2=fig.add_subplot(142,sharey=ax1)
data.Survived[data.Sex=='female'][data.Pclass==3].value_counts().plot(kind='bar',label='female,low class',color='pink')
ax2.set_xticklabels([u"获救",u"未获救"],rotation=0)
ax2.legend([u"女性/低级舱"],loc='best')

ax3=fig.add_subplot(143,sharey=ax1)
data.Survived[data.Sex=='male'][data.Pclass!=3].value_counts().plot(kind='bar',label='male highclass',color='lightblue')
ax3.set_xticklabels([u"获救",u"未获救"],rotation=0)
ax3.legend([u"男性/高级舱"],loc='best')

ax4=fig.add_subplot(144,sharey=ax1)
data.Survived[data.Sex=='male'][data.Pclass==3].value_counts().plot(kind='bar',label='male highclass',color='blue')
ax4.set_xticklabels([u"获救",u"未获救"],rotation=0)
ax4.legend([u"男性/低级舱"],loc='best')

plt.show()

#%%
#登录港口与获救情况之间的关系
data.Embarked.value_counts()
Survived_0=data.Embarked[data.Survived==0].value_counts()
Survived_1=data.Embarked[data.Survived==1].value_counts()
df=pd.DataFrame({u'未获救':Survived_0,u'获救':Survived_1})
df.plot(kind='bar',stacked=True)
plt.title(u"各登录港口乘客的获救情况")
plt.xlabel(u"登录港口")
plt.ylabel(u"人数")
plt.show()

#堂兄弟/妹，孩子/父母有几人，对是否获救的影响。
#%%
g=data_train.groupby(['SibSp','Survived'])
df=pd.DataFrame(g.count()['PassengerId'])
df

#%%
g = data_train.groupby(['Parch','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
df


#Cabin 与获救与否相关情况
# 有效值少，且分散
data.Cabin.value_counts()

# 故下一步选择有无Cabin信息这个粗粒度看看获救与否
fig=plt.figure()
fig.set(alpha=0.2)

#%%
Survived_cabin=data.Survived[pd.notnull(data.Cabin)].value_counts()
Survived_nocabin=data.Survived[pd.isnull(data.Cabin)].value_counts()

df=pd.DataFrame({u"有":Survived_cabin,u"无":Survived_nocabin}).transpose()
df.plot(kind='bar',stacked=True)
plt.title(u"按Cabin有无看获救情况")
plt.xlabel(u"Cabin有无")
plt.ylabel(u"人数")
plt.show()


#第七部分 简单数据预处理
# to do 