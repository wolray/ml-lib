from lib import *

data=loadtxt('ex2data1.txt',delimiter=',')
X=data[:,:-1]
X=add_ones(X)
y=data[:,-1]

lamb=[0,1,10,100]
t0=zeros(X.shape[1])

for i in lamb:
    J0=cost(t0,X,y,i)
    out=op.fmin(cost,t0,args=(X,y,i),maxiter=500,disp=False,full_output=True)
    t,J=out[0],out[1]
    print('\nlamb=%d' %i)
    print(J0,J)
