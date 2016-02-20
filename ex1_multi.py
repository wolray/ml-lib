from lib import *

theta_0,X,y=load_data('ex2data2.txt',1)

alpha=0.01
iters=440

t1=grad_descent(theta_0,X,y,alpha,iters)
t2=norm_eqn(X,y)
myprint((t1,t2))
