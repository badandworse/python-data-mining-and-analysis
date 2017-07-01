#matplotlib API 入门

#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import plot
import pandas as pd

plt.plot(np.arange(10))

#  Figure and Subplot
#   matplotlib 的图像都位于Figure对象
#   plt.figure创建一个新的Figure
fig=plt.figure()
#   add_subplot创建一个或多个subplot，
#   代码意思是图像应该是2*2的，且选中的是4个中的第一个，编号是从1开始
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
figfig
ax3
#  发出绘图指令,'k--'表示绘制黑色需线图


from numpy.random import randn
plt.plot(randn(50).cumsum(),'k--')
fig
ax1.hist(randn(100),bins=20,color='k',alpha=0.3)
ax2.scatter(np.arange(30),np.arange(30)+3*randn(30))
fig
fig,axes=plt.subplots(2,3)
fig
axes
axes[0,1]

#  plt.subplots 创建一个新的Figure
#  返回一个已创建的subplot对象的numpy数组
#  subplots参数:
#   nrows,ncols -- 行数,列数  
#   sharex,sharey--具有相同的X,Y轴刻度 
#   subplot_kw --用于创建subplot的关键字典
#   **fig_kw 创建figure时的其他关键字，如plt.subplots(2,2,figsize=(8,6))

#%%
fig,axes=plt.subplots(2,2,sharex=True,sharey=True)
for i in range(2):
    for j in range(2):
        axes[i,j].hist(randn(500),bins=50,color='k',alpha=0.5)
#  调整各个subplots之间的距离
plt.subplots_adjust(wspace=0,hspace=0)


#  颜色、标记和线性
ax1.plot([1,2],[2,3],'g--')
fig
# 上下两个等效，plot接受一组X和Y坐标，
# 接受一个表示颜色和线型的字符串所需
ax2.plot([1,2],[2,3],linestyle='--',color='g')
fig
ax3.plot([1,2],[2,3],linestyle='-',color='g')
fig

plt.plot(randn(30).cumsum(),'ko--')

plot(randn(30).cumsum(),color='g',linestyle='dashed',marker='o')

data=randn(30).cumsum()

plt.plot(data,'k--',label='Default')

plt.plot(data,'k-',drawstyle='steps-post',label='steps-post')
#  legend会添加一个subplot图例
plt.legend(loc='best')

#  xlim()不带参数，xlim返回当前的X轴绘图范围
plt.xlim()
#  xlim()带参数，xlim将当前的X轴范围设置为0到10
plt.xlim([0,10])

#  对于每个subplot对象上的两个方法，以xlim为例，就是subplots.get_xlim和subplots.set_xlim
#%%
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(randn(1000).cumsum())
#%%
#  set_xticks规定刻度放在数据范围的哪些位置
#  set_xticklabels将任何其他的值用作标签
ticks=ax.set_xticks([0,250,500,750,1000])
labels=ax.set_xtickslabels(['one','two','three','four','five'],rotation=30,fontsize='small')
fig
#  set_title设置一个标题
ax.set_title('my first matplotlib plot')

#  为X轴命名，
ax.set_xlabel('Stages')
fig

#  添加图例
#  在用subplot中画图时传入label参数，
#  然后使用ax.legend()自动创建图例
#  loc是告诉将图例放在哪
#%%
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(randn(1000).cumsum(),'k',label='one')
ax.plot(randn(1000).cumsum(),'k--',label='two')
ax.plot(randn(1000).cumsum(),'k.',label='three')
ax.legend(loc='best')
fig

#%%
#  注解以及在Subplot上绘图
#  text在指定坐标添加指定文本
ax.text(100,100,'hello world!',family='monospace',fontsize=10)
fig
filepath2='C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch08PlottingAndVisualzation/data'

# annotate()函数是为图形中添加箭头与对应文本来作为标准作用

#%%
from datetime import datetime
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
data=pd.read_csv(filepath2+'/spx.csv',index_col=0,parse_dates=True)
spx=data['SPX']
spx.plot(ax=ax,style='k-')
crisis_data=[
    (datetime(2007,10,11),'Peak of bull market'),
    (datetime(2008,3,13),'Bear Stearns Fails'),
    (datetime(2008,9,15),'Lehman Bankruptcy')
]
for date,label in crisis_data:
    ax.annotate(label,xy=(date,spx.asof(date)+50),
                xytext=(date,spx.asof(date)+200),
                arrowprops=dict(facecolor='black'),
                horizontalalignment='left',verticalalignment='top')

#  放大到2007--2010
ax.set_xlim(['1/1/2007','1/1/2011'])
ax.set_ylim([600,1800])
ax.set_title('Important dates in 2009-2009 financial crisis')


#  图形绘制
#%%
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
#  绘制矩形，圆，三角
rect=plt.Rectangle((0.2,0.75),0.4,0.15,color='k',alpha=0.3)
circ=plt.Circle((0.7,0.2),0.15,color='b',alpha=0.3)
pgon=plt.Polygon([[0.15,0.15],[0.35,0.4],[0.2,0.6]],color='g',alpha=0.5)
ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)

#  将图表保存到文件
plt.savefig('figpath.png',dpi=400,bbox_inches='tight')
fig
#  解决savefig后是空白的方法
#  

#%%
fig=plt.figure()
fig
ax=fig.add_subplot(1,1,1)
#  绘制矩形，圆，三角
#  先调用plt.gcf()函数取得当前绘制的figure并调用savefig函数
#  然后再调用savefig函数
rect=plt.Rectangle((0.2,0.75),0.4,0.15,color='k',alpha=0.3)
circ=plt.Circle((0.7,0.2),0.15,color='b',alpha=0.3)
pgon=plt.Polygon([[0.15,0.15],[0.35,0.4],[0.2,0.6]],color='g',alpha=0.5)
ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)
foo_fig=plt.gcf()   #'get current figure'
plt.savefig('figpath.png',dpi=400,bbox_inches='tight')

fig
#%%
#  savefig不一定非要写入磁盘，也可以写入任何文件型队形，比如BytesIO

from io import BytesIO
buffer=BytesIO()
plt.savefig(buffer,format='png')
buf.seek(0)

