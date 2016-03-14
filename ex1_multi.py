from lib import *

X,y=load_data('ex1data2.txt')
X=inorm_feature(X)
alpha=0.01
iters=[10,100,1000,5000]
n0=X.shape[1]-1
t0=zeros((n0+1,1))

t1,t2,t3,t4=map(lambda k:igrad_des(t0,X,y,alpha,k),iters)
t_group=(t1,t2,t3,t4)
t_eqn=inorm_eqn(X,y)
for i in range(len(iters)):
    print()
    print('iters=%d' % iters[i])
    print(c_[t_group[i],t_eqn])
