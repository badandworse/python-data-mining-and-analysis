import pandas as pd
import numpy as np
from matplotlib import pyplot as pl
import json

#%%
unames=['user_id','gender','age','occupation','zip']
users=pd.read_table('C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch02/data/movielens/users.dat',sep='::',header=None,names=unames,engine='python')

rnames=['user_id','movie_id','rating','timestamp']
ratings=pd.read_table('C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch02/data/movielens/ratings.dat',sep="::",header=None,names=rnames,engine='python')

mnames=['movie_id','title','genres']
movies=pd.read_table('C:/Users/xg302/git/python-data-mining-and-analysis/data analysis/ch02/data/movielens/movies.dat',sep='::',header=None,names=mnames)

#切片语法
users[:5]
ratings[:5]
movies[:5]

ratings

#将三张表进行合并，panda根据列名的重叠情况推出哪些列是合并的
data=pd.merge(pd.merge(ratings,users),movies)
data

data.ix[0]

#生成一张表，行为电影名，列为分数，男女区分开，直观对比，按性别计算每部电影的平均分
mean_ratings=data.pivot_table('rating',index='title',columns='gender',aggfunc='mean')
mean_ratings

ratings_by_title=data.groupby('title').size()


activ_titles=ratings_by_title.index[ratings_by_title>=200]
activ_titles

#选出评分数大于250的电影
mean_ratings=mean_ratings.ix[activ_titles]
mean_ratings

#%%
#对F列降序排列
top_female_ratings=mean_ratings.sort_values(by='F',ascending=False)
top_female_ratings[:10]

mean_ratings['diff']=mean_ratings['M']-mean_ratings['F']

sorted_by_diff=mean_ratings.sort_values(by='diff')
sorted_by_diff[:15]

#对行进行反序并取出前15行。切片
sorted_by_diff[::-1][:15]

ratings_std_by_title=data.groupby('title')['rating'].std()
ratings_std_by_title=ratings_std_by_title.ix[activ_titles]
ratings_std_by_title.order(ascending=False)[:10]

#%%
l=[1,2,4,4,5,56]
l[::2]