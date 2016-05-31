from lib import *

data=loadtxt('ex1data2.txt',delimiter=',')
x=data[:,:-1]
x=NormFeat(x)
x=AddOnes(x)
y=data[:,-1:]

lamb=0
alpha=0.1
iters=[1,3,10,30,100,300]
t0=zeros((x.shape[1],1))

t_eqn=NormEqn(x,y)
for i in iters:
    t=GradDes(t0,x,y,lamb,alpha,i)
    print(t0)
    print('\niters=%d' %i)
    print(c_[t,t_eqn])
