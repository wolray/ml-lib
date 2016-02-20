from lib import *

theta_0,X,y=load_data('ex2data2.txt',1)

alpha=0.01
iters=[10,100,1000,5000]

t1,t2,t3,t4=map(lambda i:grad_descent(theta_0,X,y,alpha,i),iters)
t_group=(t1,t2,t3,t4)
t_eqn=norm_eqn(X,y)
for i in range(len(iters)):
    print()
    print('iters=%d' % iters[i])
    myprint((t_group[i],t_eqn))
