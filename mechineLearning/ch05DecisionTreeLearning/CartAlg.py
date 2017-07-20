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
            GiniList.append(float(sum))
            n=n+1
        return GiniList
    else:
        group2=dt.groupby([attr1,setIndex]).size()
        group3=dt.groupby([attr1]).size()
        p1=group3/group3.sum()
        indexAttr=p1.index
        sum=0
        for i in indexAttr:
            size1=group2[i]
            #print(size1)
            p=size1/size1.sum()
            sum=sum+p1[i]*2*p[i]*(1-p[i])
        sum=Decimal(sum).quantize(Decimal('.01'),rounding=ROUND_HALF_UP) 
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
charactersIndex=dataF.columns

#%%
n=0
list1=getGini(dataF[[charactersIndex[1],charactersIndex[charactersNum]]],charactersIndex[1],charactersIndex[charactersNum])
list1