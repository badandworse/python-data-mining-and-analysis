
#%%
from matplotlib import pyplot as plt
import numpy as np

x=np.linspace(-np.pi,np.pi,256,endpoint=True)
C,S=np.cos(x),np.sin(x)
#%%
plt.plot(C)
plt.plot(S)

plt.show()


#实例化默认设置图像
#%%
#创建尺寸为8*6inches的figure，每个inch上有90个点
plt.figure(figsize=(8,6),dpi=90)

plt.subplot(1,1,1)

plt.plot(x,C,color="blue",linewidth=1.0,linestyle="-")
plt.plot(x,S,color="red",linewidth=1.0,linestyle="-")
#设定x轴的范围
plt.xlim(-4,4)
#x轴的范围
plt.xticks(np.linspace(-1,1,5,endpoint=True))
#同上
plt.ylim(-1.0,1.0)
plt.yticks(np.linspace(-1,1,5,endpoint=True))

plt.show()

#%%
#  改变线宽和颜色
plt.figure(figsize=(20,15),dpi=90)
plt.plot(x,C,color="blue",linewidth=2.5,linestyle="-")
plt.plot(x,S,color="red",linewidth=2.5,linestyle="-")

plt.xlim(x.min()*1.1,x.max()*1.1)
plt.ylim(C.min()*1.1,C.max()*1.1)

#plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
#plt.yticks([-1,0,+1])


#  使x和y轴的标签更加美观,使用latex
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
          [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

plt.yticks([-1,0,+1],[r'$-1$', r'$0$', r'$+1$'])


#  移动轴线(spines)
##Spines是连接坐标刻度和标记数据区域的线条，
##它们可以被置于图形任意位置
##现在将它们移动到中央位置
##一共有4跟线条，将top和right设置为无色，
##把bottom和left移动0坐标

ax=plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

#  添加图裂
#  并设置legend()图形左上角图例
plt.plot(x,C,color="blue",linewidth=2.5,linestyle="-",label="cosine")
plt.plot(x,S,color="red",linewidth=2.5,linestyle="-",label='sine')
plt.legend(loc='upper left')

#  标注数据点,使用annotate()函数
t = 2 * np.pi / 3
plt.plot([t, t], [0, np.cos(t)], color='blue', linewidth=2.5, linestyle="--")
plt.scatter([t, ], [np.cos(t), ], 50, color='blue')

plt.annotate(r'$sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
             xy=(t, np.sin(t)), xycoords='data',
             xytext=(+10, +30), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.plot([t, t],[0, np.sin(t)], color='red', linewidth=2.5, linestyle="--")
plt.scatter([t, ],[np.sin(t), ], 50, color='red')

plt.annotate(r'$cos(\frac{2\pi}{3})=-\frac{1}{2}$',
             xy=(t, np.cos(t)), xycoords='data',
             xytext=(-90, -50), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        
#  细节决定成败
##刻度标签因为线条的遮挡不易看清，
##通过该表字体大小和背景透明度可以使线条和标签同时看见
for label in ax.get_xticklabels()+ax.get_yticklabels():
    label.set_fontsize(16)
    label.set_bbox(dict(facecolor='white',edgecolor='None',alpha=0.65))


#条形图
#%%
n=12
X=np.arange(n)
Y1=(1-X/float(n))*np.random.uniform(0.5,1.0,n)
Y2=(1-X/float(n))*np.random.uniform(0.5,1.0,n)
plt.bar(X,+Y1,facecolor='#9999ff',edgecolor='white')
plt.bar(X,-Y2,facecolor='#ff9999',edgecolor='white')

for x,y in zip(X,Y1):
    plt.text(x+0.1,y+0.05,'%.2f' %y,ha='center',va='bottom')
plt.ylim(-1.25,+1.25)

for x,y in zip(X,Y2):
    plt.text(x+0.1,-y-0.1,'%.2f' %y,ha='center',va='bottom')

plt.xlim(-.5,n)
plt.xticks(())
plt.ylim(-1.25,1.25)
plt.yticks(())
plt.show()


# 3D plots
#%%
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
X=np.arange(-4,4,0.25)
Y=np.arange(-4,4,0.25)
X,Y=np.meshgrid(X,Y)
R=np.sqrt(X**2+Y**2)
Z=np.sin(R)

ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='hot')
ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
ax.set_zlim(-2,2)

#%%
fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X ** 2 + Y ** 2)
Z = np.sin(R)

#rstride 代表row的步长， cstride则代表columns的步长
#cmap表示的图像颜色变化
ax.plot_surface(X, Y, Z, rstride=1, cstride=10, cmap=plt.cm.hot)
ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
ax.set_zlim(-2, 2)

plt.show()
#%%
