from lib import *

data=loadtxt('ex2data2.txt',delimiter=',')
x=data[:,:-1]
x=AddOnes(x)
y=data[:,-1]

lamb=[0,1,10,100]
t0=zeros(x.shape[1])

cost0=Cost(t0,x,y,0)
for i in lamb:
    out=op.fmin(Cost,t0,args=(x,y,i),maxiter=500,disp=False,full_output=True)
    t,cost=out[0],out[1]
    print('\nlamb=%d' %i)
    print(cost0,cost)
