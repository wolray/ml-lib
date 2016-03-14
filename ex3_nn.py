from lib import *

X,y=load_mat('ex3data1.mat')
t1_trans,t2_trans=load_nn('ex3weights.mat')
n=[400,25,10]
lamb=0

t=append(t1_trans.T,t2_trans.T)

npredict(t,n,X,y)
print('ex3_nn: 97.5%')
