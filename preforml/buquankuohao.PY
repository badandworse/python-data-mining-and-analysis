#%%
def makeupP(m_str,length):
    i=0
    leftStack=[]
    ll=m_str
    while(i<length):
        if m_str[i]=='[':
            leftStack.append(m_str[i])
            i=i+1
        else:
            if len(leftStack)>0:
                leftStack.pop()
                i=i+1
            else:
                ll='['+ll
                i=i+1
    ll=ll+']'*len(leftStack)        
    return ll

#%%
makeupP('[][[][][][]',3)

