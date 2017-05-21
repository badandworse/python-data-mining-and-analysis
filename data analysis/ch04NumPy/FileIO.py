import numpy as np 

#%%
#将数组以二进制格式保存到磁盘中
#np.save 和 np.load 是读写磁盘数组数据的两个主要函数
arr=np.arange(10)
arr2=np.arange(20)
np.save('some_array',arr)
np.load('some_array.npy')

#np.savez可以将多个数组保存到一个压缩文件中，将数组以关键字参数的形式传入
#加载.npz文件时，得到一个类似字典的对象，对各个数组延迟加载
np.savez('array_archive.npz',a=arr,b=arr2)
arch=np.load('array_archive.npz')
arch['a']
arch['b']

#%%
#存取文件
#指定各种分隔符，针对特定列的转换器函数、需要跳过的行数等
#np.savetxt执行相反的操作:将数组写到以某种分隔符隔开的文本文件中
arr=np.loadtxt('array_ex.txt',delimiter=',')
arr


#genfromtxt跟loadtxt差不多，只不过面向的是结构化数组和缺失数据处理

