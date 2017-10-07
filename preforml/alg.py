
#%%
def checkio(data):
    m=dict()
    for i in data:
        if i in m:
            m[i] +=1
        else:
            m[i]=1
    return m


#%%
stringTest='aaabbaaac'
ll=checkio(stringTest)

sum=0.0
#%%
for i in ll:
    sum=sum+ll[i]
a=sum/len(ll)
print('%.2f' %a)

import re


mm=re.compile(r'((.)\2*)')
groups=mm.match('123ssss')
print(1) 