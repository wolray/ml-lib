from lib import *

X,y=load_data('ex1data1.txt')
alpha=0.01
iters=1500
n0=X.shape[1]-1
t0=zeros((n0+1,1))
lamb=0

J=icost(t0,X,y,lamb)
t=igrad_des(t0,X,y,lamb,alpha,iters)
p1=10000*hi(array([1,3.5]),t)[0]
p2=10000*hi(array([1,7]),t)[0]
print(J,p1,p2)
print('ex1: 32.0727 4519.767868 45342.450129')
