from lib_ln import *

data=loadtxt('ex1data2.txt',delimiter=',')
X=data[:,:-1]
X=norm_features(X)
X=add_ones(X)
y=data[:,-1:]

lamb=0
alpha=0.1
iters=[1,3,10,30,100,300]
t0=zeros((X.shape[1],1))

for i in iters:
    t=grad_des(t0,X,y,lamb,alpha,i)
    t_eqn=norm_eqn(X,y)
    print('\niters=%d' %i)
    print(c_[t,t_eqn])
