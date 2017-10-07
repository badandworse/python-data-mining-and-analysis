def sorted(m,length):
    result_l=sorted(m)
    i=0

    zwhile True:
        ll=m[length-1]+sorted(n[:length-2])
        i=i+1
        if ll==result_l:
            break
        ll=ll[0]+sorted(ll[1:])
        i=i+1
        if ll==result_l:
            break
    q=0
    ll=m
    while True:
        ll=ll[0]+sorted(ll[1:])
        q=q+1
        if ll==result_l:
            break
        ll=m[length-1]+sorted(n[:length-2])
        q=q+1
        if ll==result_l:
            break
    if q>i:
        return i
    else:
        return q

n=[3,2,1]
sorted(n)

[1]+n