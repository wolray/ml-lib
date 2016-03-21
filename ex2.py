from lib import *

data=loadtxt('ex2data1.txt',delimiter=',')
X=data[:,:-1]
X=add_ones(X)
y=data[:,-1:].ravel()

lamb=0
t0=zeros(X.shape[1])

J0=cost(t0,X,y,lamb)
out=op.fmin(cost,t0,args=(X,y,lamb),maxiter=500,disp=False,full_output=True)
t,J=out[0],out[1]
prob=h(array([1,45,85]),t)
print3([J0,J,prob])
print('MATLAB: 0.693 0.203 0.776')
