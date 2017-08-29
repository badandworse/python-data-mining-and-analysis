#%%
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import decimal
from decimal import Decimal,ROUND_HALF_UP


#%%
#读取数据文件，默认数据文件为经过处理的csv，
def readData(filePath,fileName):
    dt=pd.read_csv(filePath+fileName)
    return dt

#%%
# 有问题
def getGini(dt,attr1,setIndex):
    group1=dt.groupby(attr1)
    sumAttr=len(group1.size())
    Attr=group1.size().index
    GiniList=[]
    if sumAttr!=2:
        n=0
        while n<sumAttr:
            attrdt=dt.copy(deep=True)
            attrdt[attr1]=(attrdt[attr1]==Attr[n]).astype(int)
            sum=computeGini(attrdt,attr1,setIndex)
            GiniList.append(float(sum))
            n=n+1
        return GiniList
    else:
        sum=computeGini(dt,attr1,setIndex)        
        GiniList.append(float(sum))
        return GiniList

#%%
def computeGini(attrdt,attr1,setIndex):
    group2=attrdt.groupby([attr1,setIndex]).size()         
    group3=attrdt.groupby([attr1])
    p1=group3.size()/group3.size().sum()
    indexAttr=p1.index
    sum=0
    for i in indexAttr:
        size1=group2[i]
        p=size1/size1.sum()
        sum=sum+p1[i]*2*p[i]*(1-p[i])
    sum=Decimal(sum).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
    return sum
#%%
filepath="C:/Users/xg302/git/python-data-mining-and-analysis/mechineLearning/ch05DecisionTreeLearning/"
filename="data.csv"

dataF=readData(filepath,filename)

#%%
# 对dataframe指定列根据条件筛选赋值
#dataF
#dataF['age']=(dataF['age']==2).astype(int)

print(1)
charactersNum=len(dataF.columns)-1
charactersIndex=dataF.columns[:charactersNum]
charactersIndex

#%%
n=0

#%%
# 返回下个划分元素
def get_split(dataF,setIndex1,charactersIndex,minIndex='',minValue=1000):    
    valueL=[]
    for m in charactersIndex:
        valuesList=getGini(dataF[[m,setIndex1]],m,setIndex1)
        mm=valuesList.index(min(valuesList))
        valuesList.sort()
        if valuesList[0]<minValue:
            minValue=valuesList[0]
            minIndex=m
            valueL=valuesList
    if len(valueL)==1:
        b_index=dataF.groupby([minIndex]).size().index
        print(b_index)
        left_dt=dataF[dataF[minIndex]==b_index[1]]
        right_dt=dataF[dataF[minIndex]==b_index[0]]
        return {'index':minIndex,'group':(left_dt,right_dt)}
    else:
        #即所有数据分类是多结果离散
        b_index=dataF.groupby([minIndex]).size().index
        dataF[minIndex]=(dataF[minIndex]==b_index[mm]).astype(int)
        newIndex=str(b_index[mm])+'('+minIndex+')' 
        dataF=dataF.rename(columns={minIndex:newIndex})
        left_dt=dataF[dataF[newIndex]==1]
        right_dt=dataF[dataF[newIndex]==0]
        return {'index':newIndex,'group':(left_dt,right_dt)}
      
                     


#node=get_split(dataF[['age',dataF.columns[4]]],dataF.columns[4],[charactersIndex[0]])
#node=get_split(dataF,dataF.columns[4],charactersIndex)
#node


#%%
def split(node,max_depth,min_size,depth):
    


#%%
#todo : 树的生成未完成，有点没有头绪，决定补下算法
def build_tree(data,max_depth,min_size):
    allIndex=data.columns
    charactersIndex=allIndex[:len(allIndex)-1]
    typeIndex=allIndex[len(allIndex)-1]
    root=get_split(data,typeIndex,charactersIndex)
    split(root,max_depth,min_size,1)


#%%
# data,max_depth,min_size
tree=build_tree(dataF,3,1)

#%%
ll=[3,45,2]
min(ll)
ll.index(2)
sortedll=ll.sort()
