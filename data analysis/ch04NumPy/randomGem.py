import numpy as np
from numpy import random
from numpy import size
from random import normalvariate
#random中有函数能高效生成多种概念分布的样本值函数

#normal 得到一个标准正态分布的4*4样本
samples=np.random.normal(size=(4,4))
samples


N=10000


#%%
#随机漫步 纯python版本

import random
import matplotlib.pyplot as plt
position=0
walk=[position]
steps=1000
for i in range(steps):
    step=1 if random.randint(0,1) else -1
    position+=step
    walk.append(position)

plt.plot(walk)


#%%
#numpy版本
nsteps=1000
draws=np.random.randint(0,2,size=nsteps)
steps=np.where(draws>0,1,-1)
walk=steps.cumsum()
plt.plot(walk)

walk.min()
walk.max()

#argmax返回第一个最大值的索引
(np.abs(walk)>=10).argmax()

#%%
#一次模拟多个随机漫步
nwalks=5000
nsteps=1000
draws=np.random.randint(0,2,size=(nwalks,nsteps))
np.size(draws,0)
steps=np.where(draws>0,1,-1)
walks=steps.cumsum(1)
walks
np.size(walks,1)

walks.max()
walks.min()

hits30=(np.abs(walks)>=30).any(1)
hits30

hits30.sum() #到达30或者-30的数量

crossing_times=(np.abs(walks[hits30])>=30).argmax(1)
crossing_times.mean()

#%%
testarr=np.array([[1,2,3,4],[3,4,5,6]])
testarr
np.size(testarr,1)