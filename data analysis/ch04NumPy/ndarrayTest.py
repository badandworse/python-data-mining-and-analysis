import numpy as np

#np.array函数接受一切序列型对象(包括其他数组),然后产生一个新的含有传入数据的NumPy数组
data1=[6,7,8,1.0,0]
arr1=np.array(data1)
arr1

#嵌套序列（比如由一组等长列表组成的列表）将会被转换为一个多维的uuzu
data2=[[i for i in range(4)],[n for n in range(5,9)]]
data2
#%%
arr2=np.array(data2)
arr2
#arr2是几维的
arr2.ndim
arr2.shape

'''
numpy 中几种创建ndarray对象的函数:
zeros,zeros_like 根据指定形状和dtype创建一个全是0的数组，zeros_like以另一个数组为参数，并根据其
                形状和dtype创建一个全0的数组
ones、ones_like 同上
empty empty_like 创建新的数组，只分配内存空间而不填充任何值
eye identity  return N*N Unit matrix（对角为1，其余为0）

如果没有特别指定，数据类型基本都是float64
'''
#创建一个全为0的ndarray
np.zeros(10)
#创建一个3*6 全为0的ndarray
np.zeros((3,6))

#返回的ndarray的值都是未被初始化的值
np.empty((2,3,2))

np.eye(5)



'''
numpy数据类型，dtype可以得到ndarray的数据类型，
astype则可以转换数据类型,产生一个新的ndarray
float to int  小数部分将会被截断
'''
arr3=np.array([1,3,4],dtype=np.float64)
arr3.dtype
arr4=arr3.astype(np.int32)
arr4.dtype

arr5=np.array([3.7,4.6,3.5])
arr5.dtype
arr6=arr5.astype(np.int64)
arr6

#%%
int_array=np.arange(10)
calibers=np.array([.22,.26,.356,.50])
int_array.astype(calibers.dtype)

empty_uint32=np.empty(8,dtype='u4')
empty_uint32


'''
ndarray的运算: 大小相等之间的ndarray之间的任何算术运算都会将运算应用到元素级
数组与标量之间的算术运算也会将那个标量值传播到各个元素
'''

arr7=np.array([[1.,2.,3.],[4.,5.,6.]])
arr7
arr7*arr7
arr7-arr7
1/arr7


#%%
#基本的索引与切片
arr=np.arange(10)
#一个标量值赋值给一个切片时，该值会自动传播到整个选区，跟列表最重要的区别在于，数组切片是原始数组的视图
#这意味着数据不会被复制，视图上的任何修改都会被直接反映到源数组上
#如果想得到ndarray的切片的一份副本而非视图，就需要显式地进行复制操作，例如arr[5:8].copy()
arr[5:8]=12
arr
arr_slice=arr[5:8]
arr_slice[1]=123456
arr
arr_slice[:]=64
arr

arr2d=np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2d[2]

arr2d[0][2]
arr2d[0,2]

#在多维数组中，如果省略了后面的索引，则返回对象会是一个纬度低一点的ndarray,利用索引选取的数组子集中返回的数组都是视图
arr3d=np.array([[[1,2,3],[4,5,6]],[[7,8,8],[10,11,12]]])
arr3d
arr3d[0]


#切片索引
arr2d
arr2d[:2,1:]
#整数索引和切片混合，得到低纬度的切片,切片是视图， 因此对其赋值会扩散到整个选区
arr2d[1,:2]=0
#‘只有冒号’表示选取整个轴
arr2d[:,:1]
arr2d

#%%
#布尔索引，通过布尔型索引选数组中的数据
#多个布尔条件,使用&、|
names=np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
data=np.random.randn(7,4)
names=='Bob'
data

data[names=='Bob']=0
data
data[names=='Bob',2:]

#%%
#花式索引,利用整数数组进行索引
arr8=np.empty((8,4))
for i in range(8):
    arr8[i]=i

arr8

arr8[[4,3,0,6]]

arr9=np.arange(32).reshape((8,4))
arr9
#最终选出的元素为(1,0),(5,3),(7,1),(0,2)
arr9[[1,5,7,0],[0,3,1,2]]
arr9[[1,5,7,0]][:,[0,3,1,2]]

#np.ix_将两个一维整数数组转换为一个用于选取方形区域的索引器
arr9[np.ix_([1,5,7,0],[0,3,1,2])]=0
arr9

#%%
#转置是重塑的一种特殊形式，它返回的是源数据的视图,ndarray.T进行简单的轴转换
#如多维ndarray需要使用transpose函数该函数得到一个由轴编号组成的元祖才能对这些轴进行转置


arr10=np.arange(15).reshape((3,5))
arr10
arr10.T

#%%
#利用np.dot计算矩阵内积X^T X
arr=np.random.randn(6,3)
np.dot(arr.T,arr)

#ndarray还有一个swapaxes方法，它需要接受一对轴编号
arr=np.arange(16).reshape(2,2,4)
arr
arr.transpose((1,0,2))
arr.swapaxes(1,2)



