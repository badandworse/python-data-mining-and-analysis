import numpy as np
from scipy import io as spio
from scipy import misc
import scipy.linalg as la
from matplotlib import pyplot


# Scipy.special模块包括大量的贝塞尔函数
# 这里我们将使用函数JN和YN，它们是贝塞尔函数 
# 的第一和第二类及实值量级。我们还包括了 
# 函数jn_zeros和yn_zeros，给JN和YN函数赋零值
from scipy.special import jn,yn,jn_zeros,yn_zeros

np.lookfor('sum')
a=np.ones((3,3))
a
#载入和保存matlab文件
spio.savemat('file.mat',{'a':a}) #savemat expects a dictionary
data=spio.loadmat('file.mat',struct_as_record=True)
data['a']

#读取图像
misc.imread('figpath.png')

#scipy.special
#超越函数,特殊函数
#%%
n=0
x=0.0
print("J_%d(%f)=%f" %(n,x,jn(n,x)))

x=1.0
print("Y_%d(%f)=%f"%(n,x,yn(n,x)))
#%%
x=np.linspace(0,10,100)
fig,ax=pyplot.subplots()

for n in range(4):
    ax.plot(x,jn(n,x),label=r"$J_%d(x)$" %n)
ax.legend();


#线性代数操作:scipy.linalg

arr=np.array([[1,2],[3,4]])
#计算行列式
la.det(arr)

#scipy.linalg.inv()计算方阵的逆矩阵
arr=np.array([[1,2],[3,4]])
iarr=la.inv(arr)
iarr

#计算奇异矩阵(行列式为0)的逆矩阵时会报错，因为不存在
arr=np.array([[3,2],[6,4]])
iarr=la.inv(arr)

#奇异值计算
arr=np.arange(9).reshape((3,3))+np.diag([1,0,1])
uarr,spec,vharr=la.svd(arr)
uarr
#M=UΣV* M为m*n阶矩阵
#U是m*m阶矩阵，Σ是m×n阶非负实数对角矩阵
#而V*，即V的共轭转置
#spec就是奇异值
#urarr为U，vharr为v
spec
vharr


#原始的矩阵可以使用svd的输出结果和np.dot的乘积重新生成
sarr=np.diag(spec)
sarr
svd_mat=uarr.dot(sarr).dot(vharr)
#判断两个矩阵各个元素是否足够的接近，如果是，则返回true
np.allclose(svd_mat,arr)
svd_mat
arr

#快速傅立叶变换:scipy.fftpack
##可用于降噪,未处理
from scipy import fftpack
#噪声信号的例子
#%%
time_step=0.02
period=5.
time_vec=np.arange(0,20,time_step)
#%%
sig = np.sin(2 * np.pi / period * time_vec) + \
        0.5 * np.random.randn(time_vec.size)
sig

time_vec.size
sig.size
#生成采样频率
#%%
sample_freq=fftpack.fftfreq(sig.size,d=time_step)
sig_fft=fftpack.fft(sig)
pidxs=np.where(sample_freq>0)
freqs=sample_freq[pidxs]
power=np.abs(sig_fft)
freq=freqs[power.argmax()]
np.allclose(freq,1./period)

import pylab as plt
#%%
#优化和拟合:scipy.optimize
from scipy import optimize
def f(x):
    return x**2+10*np.sin(x)
x=np.arange(-10,10,0.1)
plt.plot(x,f(x))
#从图中可以看到，此函数全局最小为-1.3，局部最小有个3.8
#常用的求解此函数最小值的方法是确定初始点，然后执行梯度下降算法
optimize.fmin_bfgs(f,0)
#梯度下降的缺陷在于有时候可能会被困在一个局部最小值，而得不到全局的最小值
#这取决与初始点的选取
##如果初始点选在3，则会选取3.8左右的那个局部最小点作为返回
optimize.fmin_bfgs(f,3)

##scipy.optimize.basinhopping()包含
# 一个求解局部最优值的算法
# 和有一个为该算法提供随机初始点的函数

optimize.basinhopping(f,0)

optimize.brute(f,10)

# scipy中包含许多全局最优化的算法

# 使用fminbound()寻找指定区间内的局部最小值
xmin_local=optimize.fminbound(f,0,10)
xmin_local

# 使用brute()找到全局最优解
grid=(-10,10,0.1)
xmin_global=optimize.brute(f,(grid,))

# 利用 fsolve()寻找标量函数的零点
root=optimize.fsolve(f,1) #预测一个零点在1附近 
root
root2=optimize.fsolve(f,-2.5)
root2

#曲线拟合
xdata=np.linspace(-10,10,num=20)
xdata.size
ydata=f(xdata)+np.random.randn(xdata.size)
ydata
#%%
def f2(x,a,b):
    return a*x**2+b*np.sin(x)
guess=[2,2]
parmas,parmas_covariance=optimize.curve_fit(f2,xdata,ydata,guess)
parmas[1]
parmas_covariance

#%%
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(x,f(x),'b-',label="f(x)")
ax.plot(x,f2(x,*parmas),'r--',label="Curve fit result")
xmins=np.array([xmin_global[0],xmin_local])
ax.plot(xmins,f(xmins),'go',label="Minima")
roots=np.array([root,root2])
ax.plot(roots,f(roots),'kv',label="Roots")
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('f(x)')


#%%
def sixhump(x):
    return (4 - 2.1*x[0]**2 + x[0]**4 / 3.) * x[0]**2 + x[0] * x[1] + (-4 + \
        4*x[1]**2) * x[1] **2

x = np.linspace(-2, 2)
y = np.linspace(-1, 1)
xg, yg = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xg, yg, sixhump([xg, yg]), rstride=1, cstride=1,
                       cmap=plt.cm.jet, linewidth=0, antialiased=False)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('Six-hump Camelback function')
optimize.fmin_bfgs(sixhump,(-1,-1))
optimize.brute(sixhump,((-2,2,0.05),(-1,1,0.01)))


#统计和随机数:scipy.stats
#包含一些统计和随机过程相关的工具

a=np.random.normal(size=1000)
bins=np.arange(-4,5)
# 为true，结果在该区间的概率值，为false，
# 则为在该区间的个数
histogram=np.histogram(a,bins=bins,normed=True)[0]
histogram
histogram1=np.histogram(a,bins=bins,normed=False)[0]
histogram1
#%%
from scipy import stats
b=stats.norm.pdf(bins)
#%%
bins=0.5*(bins[1:]+bins[:-1])
plt.plot(bins,histogram1)

plt.plot(bins,b)
bins

loc,std=stats.norm.fit(a)
loc
std

# gamma分布
a=np.random.gamma(1,size=1000)
# 返回3个值，第一个是shape，第二个是loc，第3个是scale
stats.gamma.fit(a)

a[a>0].size

a.max()
bins=np.arange(0,7)
histogram_gama=np.histogram(a,bins,normed=True)[0]
histogram_gama.size
bins.size
bins=0.5*(bins[1:]+bins[:-1])
plt.plot(bins,histogram_gama)
# 给定区间与形状，得到相应的概率密度
# 然后画出来
# pdf means Probability density function.
b=stats.gamma.pdf(bins,1)
b.
plt.plot(bins,b)

#百分位数
# 中位数
np.median(a)
# 也叫第50百分位数
stats.scoreatpercentile(a,50)
# 第90百分位数
stats.scoreatpercentile(a,90)


#统计检验
#%%
a=np.random.normal(0,1,size=100)
b=np.random.normal(1,1,size=10)
c=np.random.normal(0,1,size=1000)
# 如果equal_var为false则进行的是welch's t-test
# 为true则进行独立的双样本t检验
# 返回两个值，
# 一个static The calculated t-statistic.
# 一个 P值,两个过程相同的概率，如果其值接近1，则两个过程几乎可以确定是相同的
# 如果其值接近0，那么两者可能拥有不同的均值
stats.ttest_ind(a,c,equal_var=False)





#插值计算 scipy.interpolate
# 插值计算通过在给定点中间以某种函数插入点，来估计未知函数
# 该模块在拟合实验数据并估计未知点数数值方面非常有用
# 1e-1 means 10-5. 
measured_time=np.linspace(0,1,10)
noise = (np.random.random(10)*2 - 1) * 1e-1

measures=np.sin(2*np.pi*measured_time)+noise

:class:`scipy.interpolate.interpld` 
#类可以创建一个线性插值函数
from scipy.interpolate import interp1d
# 产生一个线性插值函数
linear_interp=interp1d(measured_time,measures)

:obj:`scipy.interpolate.linear_interp` 实例可以在需要的时候获取某些值

computed_time=np.linspace(0,1,50)
linear_result=linear_interp(computed_time)

# 三次插值函数可通过kind关键字参数得到
#%%
cubic_interp=interp1d(measured_time,measures,kind='cubic')
cubic_result=cubic_interp(computed_time)

#%%
#画出的折线图
plt.plot(computed_time,linear_result,color="red",linewidth=2.5,label="linear interp")
plt.plot(computed_time,linear_result,color="orange",label="cubic interp")
# 画出的的是散点图
plt.plot(measured_time,measures,'o',color="black",ms=6,label="measures")
plt.legend(loc='upper right')

plt.scatter(measured_time,measures)

