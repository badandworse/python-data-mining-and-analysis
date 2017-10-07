import re
print(1)
def checkType(n,listString):
    ss='^[0-9].*'
    ss2='(\d*)(\w*)$'
    p1= re.compile("[a-z]+");
    p2= re.compile("[A-Z]+");
    p3= re.compile("[0-9]+");
    test1=re.compile(ss)
    test2=re.compile(ss2)
    i=0
    while i<n:
        if len(listString[i])<8:
            print('NO')
            i=i+1
            continue
        elif test1.match(listString[i]):
            print('NO')
            i=i+1
            continue
        elif test2.match(listString[i]) is None:
            print('NO')
            i=i+1
            continue
        elif p1.compile(listString[i]) is None and p2.compile(listString[i]) is None:
            print('NO')
            i=i+1
            continue
        elif p1.compile(listString[i]) is None and p3.compile(listString[i]) is None:
            print('NO')
            i=i+1
            continue
        elif p2.compile(listString[i]) is None and p3.compile(listString[i]) is None:
            print('NO')
            i=i+1
            continue
        else:
            print('YES')
            i=i+1
            continue
            

#%%


p1= re.compile("[a-z]+");
p2= re.compile("[A-Z]+");
p3= re.compile("[0-9]+");