import numpy as np
from numpy import arange,array,random,floor,transpose
from numpy import resize,sin
import scipy

a=arange(15).reshape(3,5)
a.shape
a.ndim   #array的纬度，此处为2
a.dtype  #元素类型
a.itemsize  #每个元素的需要多少位的存储空间,此处为int32,故为4

arange(18).reshape(2,3,3).ndim

# 可通过传入array()list 或者 tuple
# 直接创建
a=array((1,2))
a

c = array( [ [1,2], [3,4] ], dtype=complex )
c

##从10开始直到30，间隔为5创建array
arange(10,30,5)

##但arange用于小数
##这样产生数据时存在精度问提 
##此时用linespace跟好
##标明在0-2之间等间隔产生9个数
np.linspace(0,2,9)

#运算
#使用运算符*，则是各个位置对应的元素相乘
#而要进行矩阵乘法，则需使用np.dot()


#array一些常见的操作
b=arange(12).reshape(3,4)
b
b.sum(axis=0)
b.min(axis=0)  #min of each column

b.cumsum(axis=0)  # cumulative sum along each column

B=arange(3)
B
np.exp(B) #The exponential function is e^x 

a=arange(10)**3
a
a[:6:2]

a[::-1]
#%%
for i in a:
    print(i**(1/3.))

#%%
def f(i,j):
    return i**2+j**2
#标明将[0,1,2]X[0,1,2]组合成9组组合，带入函数，返回array
np.fromfunction(f,(3,3))  

#%%
def f(x,y):
    return 10*x+y

b=np.fromfunction(f,(5,4),dtype=int)
b
b[0:5,1] #each row in the second column of b
b[:,1]  #equivalent to the previous example

b[1:3,]  # each column in the second and third row of b
b[-1]  # last row


#shape manipulation
#floor 返回最接近给定数的整数即小于给定数
a=floor(10*random.random((3,4)))
a
a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
np.floor(a)

a.ravel()  #flatten the array 将多维变为一维
a.transpose() #转置矩阵
a.reshape(2,-1)  #如果reshape的一个参数给定的是-1，则会根据另一个自动计算
a.resize((2,6))  #resize 直接修改array本身


#stacking together different arrays
#将几个不同的arrays堆砌到一起
a=floor(10*random.random((2,2)))
a
b=floor(10*random.random((2,2)))
b
np.vstack((a,b)) #按行链接
np.hstack((a,b)) #按列链接
np.column_stack((a,b))

np.r_[1:4,0,4]
#deep copy
b=a.copy()
a is b

#indexing with arrays of indices
a=arange(12)**2
a
i=array([1,1,3,8,5])
a[i]
j=array([[3,4],[9,7]])
a[j]

#%%
palette=array([[0,0,0], #black
              [255,0,0],#red
              [0,255,0],#green
              [0,0,255],#blue
              [255,255,255]]#white
              )

image=array([[0,1,2,0],[0,3,4,0]])            
palette[image]
data = sin(arange(20)).reshape(5,4)
data
time=np.linspace(20,145,5)
time

ind=data.argmax(axis=0) #获得每行最大值的索引
ind

time_max=time[ind]
time_max

#ix_() function
a=array([2,3,4,5])
b=array([8,5,4])
c=array([5,4,6,8,3])
ax,bx,cx=np.ix_(a,b,c)
ax
bx
cx
ax.shape

mm=np.ix_(a,b,c)

result=ax+bx*cx
result
result.shape
