import numpy as np


'''
这是感知机对偶算法的代码
利用numpy提前算好内积，到时候直接调用省去很多计算
'''
#%%
def perDuality(a,b,np1,np2,np3,l):
    while True:
        i=0
        
        sumq=0
        while i<l:
            sum=0
            test=0
            q=0
            while q<l:
                sum=sum+a[q]*np2[q]*np3[i][q]
                q=q+1
            test=(sum+b)*np2[i]
            if test<=0:
                a[i]=a[i]+1
                b=b+np2[i]
                break
            else:
                sumq=sumq+1
                i=i+1
        if sumq==l:
            break
    return a,b

#%%
np1=np.array([[3,3],[4,3],[1,1]])
np2=np.array([1,1,-1])
a=np.array([0,0,0])
b=0
l=3
np3=np.dot(np1,np1.T)
np3
np3[0][0]
a,b=perDuality(a,b,np1,np2,np3,l)
print(a,b)
print(1)