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

# 对dataframe指定列根据条件筛选赋值
#dataF
#dataF['age']=(dataF['age']==2).astype(int)

charactersNum=len(dataF.columns)-1
charactersIndex=dataF.columns[:charactersNum]
charactersIndex
ll=[4,2,3]
ll.sort()
ll
#%%
n=0

#%%
setIndex1=dataF.columns[len(dataF.columns)-1]
minIndex=''
minValue=10000
setIndex1
while True:
    for m in charactersIndex:
        valuesList=getGini(dataF[[m,setIndex1]],m,setIndex1)
        valuesList.sort()
        if valuesList[0]<minValue:
            minValue=valuesList[0]
            minIndex=m
    break

minIndex
minValue


#todo : 树的生成未完成，有点没有头绪，决定补下算法