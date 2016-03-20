from lib_lg import *

data=loadtxt('ex2data1.txt',delimiter=',')
X=data[:,:-1]
X=add_ones(X)
y=data[:,-1:]

n0=X.shape[1]-1
lamb=[0,1,10,100]
t0=zeros((n0+1,1))

for i in lamb:
    J0=cost(t0,X,y,i)
    t,J=cfmin(t0,X,y,i)
    print('\nlamb=%d' %i)
    print(J0,J)
