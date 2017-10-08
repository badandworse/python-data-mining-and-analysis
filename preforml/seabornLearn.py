#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from pandas import DataFrame

np.random.seed(sum(map(ord,"aesthetics")))

def sinplot(flip=1):
    x=np.linspace(0,14,100)
    for i in range(1,7):
        plt.plot(x,np.sin(x+i*.5)*(7-i)*flip)

sinplot()
sns.set()
sinplot()


#seaborn.factorplot
dt=DataFrame({'1':[1,2,3,1,2,3,1,2,3],'2':[3,4,5,4,5,6,7,8,9]})
group=dt.groupby('1')
group.describe()
sns.factorplot('1','2',data=dt,size=4,aspect=3,kind='box')
print(1)


#1 seaborn 入门 distplot 与 kdeplot
#displot()集合了matplotlib的hist()与核函数估计kdeplot功能,
#增加了rugplot分布观测条显示与利用scipy库fit拟合参数分布的新颖用途。
#重要参数:
#hist:控制是否显示条形图
#kde:控制是否显示核密度估计图
#fit:random variable object, 可选  控制拟合的参数分布图形
from scipy import stats, integrate
np.random.seed(sum(map(ord,"distributions")))
x=np.random.normal(size=100)
sns.distplot(x,kde=False,rug=True)
sns.distplot(x,kde=False,rug=True,bins=40)

#核函数用于估计密度函数
sns.distplot(x,kde=True,hist=False,rug=True)
sns.kdeplot(x,shade=True)

#fit拟合参数分布
x=np.random.gamma(6,size=200) #生成gamma分布的数据
sns.distplot(x,kde=False,fit=stats.gamma) #fit拟合

#%%
#practice for displot
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='white',palette="muted",color_codes=True)
rs=np.random.RandomState(10)

#set up the matplotlib figure
f,axes=plt.subplots(2,2,figsize=(4,4),sharex=True)
sns.despine(left=True)

#generate a random univariate dataset
d=rs.normal(size=100)

#Plot a simple histogram with binsize determined automatically
sns.distplot(d,kde=False,color='b',ax=axes[0,0])

#Plot a kernel density estimate and rug plot
sns.distplot(d,hist=False,rug=True, color='r',ax=axes[0,1])

#Plot a filled kernel density estimate
sns.distplot(d,hist=False,color='g',kde_kws={"shade":True},ax=axes[1,0])

#Plot a historgram and kernel density estimate
sns.distplot(d,color='m',ax=axes[1,1])

plt.setp(axes,yticks=[])
plt.tight_layout()


#2barplot--条形图 countplot--计数图

#barplot主要参数:
#x,y,hue:设置x,y以及颜色控制变量
#data:设置输入数据集
#order,hue_order:控制变量绘图的顺序
#estimator:设置对每类变量的计算函数，默认为平均值，可修改为max,median,max
#ax:设置子图位置
#orient:控制绘图方向，水平或竖直
#capsize：float,optional 设置误差棒帽条的宽度

#%%
import seaborn as sns
sns.set_style("whitegrid")
tips=sns.load_dataset("tips")

ax=sns.barplot(x='day',y='total_bill',hue='sex',hue_order=['Female','Male'],data=tips)
ax = sns.barplot(x="day", y="tip", data=tips, estimator=np.median)

#%%
#countplot计数图
sns.set(style='darkgrid')
titanic=sns.load_dataset('titanic')
ax=sns.countplot(x='class',hue='who',data=titanic)

#%%
sns.set(style="white") #设置绘图背景

# Load the example planets dataset
planets = sns.load_dataset("planets")

# Make a range of years to show categories with no observations
years = np.arange(2000, 2015) #生成2000-2015连续整数

# Draw a count plot to show the number of planets discovered each year
#选择调色板，绘图参数与顺序，factorplot一种分类变量的集合作图方式，利用kind选择bar、plot等绘图参数以后将具体介绍
g = sns.factorplot(x="year", data=planets, kind="count",
                   palette="BuPu", size=6, aspect=1.5, order=years)
g.set_xticklabels(step=2) #设置x轴标签间距为2年



#3Boxplot and Violinplot
#boxplot箱线图：
#最左端线的端点是最小值 
#箱左端四分位数
#箱中间中位数
#箱右端四分位数
#线段右端最大值
#x,y,hue,order,hue_order:同上

#%%
filepath='c:/Users/xg302/git/python-data-mining-and-analysis/preforml/data/'
sns.set_style('whitegrid')
ax=sns.boxplot(x='day',y='total_bill',hue='smoker',data=tips,palette='Set3')

#for practice
dt=pd.read_csv(filepath+'Pokemon.csv')
dt.info()
dt.head()
dt=dt.drop(['Total','#','Generation','Legendary'],1)
sns.boxplot(y='Type 1',x='HP',data=dt)
sns.boxplot(y='Type 1',x='Speed',data=dt)

dt.groupby('Type 1').describe()


#Violinplot琴形图
#Violinplot结合了箱线图与核密度估计图的特点，
#它表征了在一个或多个分类变量情况下，
#连续变量数据的分布并进行了比较，
#它是一种观察多个数据分布有效方法。
#split:bool,optional #琴形图是否从中间分开两部分
#scale:{“area”, “count”, “width”}, optional #用于调整琴形图的宽带。
#area——每个琴图拥有相同的面域；
#count——根据样本数量来调节宽度；
#width——每个琴图则拥有相同的宽度。
#inner:{“box”, “quartile”, “point”, “stick”, None}, optional 
#控制琴图内部数据点的形态。
# box——绘制微型boxplot；
# quartiles——绘制四分位的分布；
# point/stick——绘制点或小竖条。
dt.info()
sns.violinplot(y='Attack',x='Generation',data=dt,hue='Legendary',palette='Set3',split=True,scale='width',inner='stick',scale_hue=False)

sns.set_style('whitegrid')#调整背景为白底
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))#由于变量过多，调整下图表大小
#观察共计与防御分布如何
ax1=sns.violinplot(x='Type 1',y='Attack',data=dt,scale='width',palette='Set3')

#%%
plt.figure(figsize=(12,6))#由于变量过多，调整下图表大小
ax1=sns.violinplot(x='Type 1',y='Defense',data=dt,scale='width',palette='Set3')

dt[dt['Name']=='Pikachu']
sns.boxplot(data=dt[dt['Type 1']=='Electric'])


#4 Implot 回归模型
#lmplot是一种集合基础绘图与基于数据建立回归模型的绘图方法。
#旨在创建一个方便拟合数据集回归模型的绘图方法，
#利用'hue'、'col'、'row'参数来控制绘图变量。
#参数:
#hue, col, row : strings #定义数据子集的变量，并在不同的图像子集中绘制
#size:scalar,optional #定义子图的高度
#markers:matplotlib marker code or list of marker codes, optional #定义散点的图标
#col_wrap:int,optional　#设置每行子图数量
#order:int,optional #多项式回归，设定指数
#logistic : bool, optional #逻辑回归
#logx : bool, optional #转化为log(x)

#Senior Example I for Practice
#%%
sns.set_style('whitegrid')
tips=sns.load_dataset('tips')

g=sns.lmplot(x='total_bill',y='tip',hue='smoker',data=tips,palette='Set1')

g = sns.lmplot(x="total_bill", y="tip", col="day", hue="day",data=tips, col_wrap=2, size=3)
