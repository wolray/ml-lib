from lib_ln import *

data=loadtxt('ex1data2.txt',delimiter=',')
X=data[:,:-1]
n0=X.shape[1]
X=norm_features(X)
y=data[:,-1:]

lamb=0
alpha=0.01
iters=[10,100,1000,5000]
t0=zeros((n0+1,1))

t1,t2,t3,t4=map(lambda k:grad_des(t0,X,y,lamb,alpha,k),iters)
t_group=(t1,t2,t3,t4)
t_eqn=norm_eqn(X,y)
for i in range(len(iters)):
    print()
    print('iters=%d' %iters[i])
    print(c_[t_group[i],t_eqn])
