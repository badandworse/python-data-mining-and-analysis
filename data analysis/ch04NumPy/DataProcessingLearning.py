import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
points=np.arange(-5,5,0.01) #1000个间隔相等的点
xs,ys=np.meshgrid(points,points)
ys
xs

z=np.sqrt(xs**2+ys**2)
z
#%%
plt.imshow(z,cmap=plt.cm.gray)
plt.colorbar()
plt.title('Image plot of $\sqrt{x^2+y^2}$ for a grid of values')


#%%
#将条件逻辑表达为数组运算
#numpy.where 函数是三元表达式x if condition else y的矢量化版本

xarr=np.array([1.1,1.2,1.3,1.4,1.5])
yarr=np.array([2.1,2.2,2.3,2.4,2.5])
cond=np.array([True,False,True,True,False])

result=[(x if c else y) for x,y,c in zip(xarr,yarr,cond)]
result

resultWithNP=np.where(cond,xarr,yarr)
resultWithNP

arr=np.random.randn(4,4)
arr
#一个数组中，所有正值替换为2，所有负值为-2
np.where(arr>0,2,-2)


#可以利用where表述出更复杂的逻辑，两个布尔型数组cond1和cond2
#根据4种不同的布尔值合实现不同的赋值操作
#np.where(cond1&cond2,0,np.where(cond1,1,np.where(cond2,2,3)))

#数学和统计方法
arr=np.random.randn(5,4) #正态分布数据
arr.mean()
np.mean(arr)

arr.sum()

#mean和sum这类的函数可以接受一个axis参数（用于计算该轴向上的统计值）
#最终结果是一个少一维的数组

#cumsum所有或指定轴的累积和
#cumprod所有货指定轴的累积积
#argmin、argmax 最大和最小元素的索引
arr.mean(axis=1)
arr.sum(1)

arr=np.array([[0,1,2],[4,5,6],[7,8,9]])
arr.cumsum(0)
arr.cumprod(1)
arr.cumprod(0)
arr.argmin()
arr.argmax()


#用于布尔型数组的方法
arr=np.random.randn(100)
(arr>0).sum()

#any 检测数组中是否存在一个或多个True
#all则检测数组中所有值是否都是True
bools=np.array([False,False,True,False])
bools.any()
bools.all()

#%%
#排序
arr=randn(8)
arr
arr.sort()
arr

arr=randn(5,3)
arr
arr.sort(1)
arr

#计算数组分位数最简单的办法是对其进行排序，然后选取特定位置的值
large_arr=randn(1000)
large_arr.sort()
large_arr[int(0.05*len(large_arr))] #5%分位数


#%%
#唯一化以及其他的集合合集


names=np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
#np.unique 找出数组中的唯一值，并返回已排序的结果
np.unique(names)

ints=np.array([3,4,2,3,3,2,3,4,5])
np.unique(ints)
#np.in1d测试一个数组中的值在另外一个数组中的成员资格，
#返回一个bool型的数组
#intersect1d(x,y) x,y共有元素，返回一个有序结果
#union1d(x,y)并集，并返回有序结果
#setdiff1d(x,y)集合的差，即元素在x中且并不在y中
#setxor1d(x,y) 集合的对称差，即存在与一个数组中但不同时存在于两个数组中的元素

values=np.array([6,0,0,3,2,5,6])
np.in1d(values,[2,3,4])
