from lib_ln import *

X,y=load_data('ex2data2.txt')
alpha=0.01
iters=[10,100,1000,5000]
n0=X.shape[1]-1
t0=zeros((n0+1,1))

t1,t2,t3,t4=map(lambda i:grad_des(t0,X,y,alpha,i),iters)
t_group=(t1,t2,t3,t4)
t_eqn=norm_eqn(X,y)
for i in range(len(iters)):
    print()
    print('iters=%d' % iters[i])
    print(c_[t_group[i],t_eqn])
