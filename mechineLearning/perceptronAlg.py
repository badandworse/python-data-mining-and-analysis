import numpy as np

'''
函数思想
'''

#%%
def getPerceptron(w,b,n1,l1,n2,l2,q):
    ll=True
    while ll:
        sum=0
        for m in n1:
            #print('m',m)
            #print('dot',np.dot(w,m.T))
            g=np.dot(w,m.T)+b
            #print (g)
            if g>0:
                sum=sum+1
            else:
                w=w+m
                b=b+1
                print(w,b)
                break
        if sum==l1:
            for m in n2:
                print('m',m)
                print('b',b)
                print('dot',np.dot(w,m.T))
                g=np.dot(w,m.T)+b
                print(g)
                if g<0:
                    sum=sum+1
                else:
                    w=w+m*(-1)
                    b=b-1
                    print(w,b)                
                    break
                
        if sum==(l1+l2):
            ll=False
    return w,b
                
   



#%%
#初始化
np1=np.array([[3,3],[4,3]])
np2=np.array([[1,1]])

w=np.array([[0,0]])
b=0
q=1


#np.dot(w,np1[0].T)
#学习率
#array使用* 是元素级的积。而要得到矩阵点积
#使用numpy.dot是一个选择。
#np1[0].T*np1[0]
#np.dot(np1[0],np1[0].T)

w,b=getPerceptron(w,b,np1,2,np2,1,1)
print('f',w,b)
#print(w,b)