from lib import *

X,y=load_data('ex1data1.txt')
alpha=0.01
iters=1500
n0=X.shape[1]-1
t0=zeros((n0+1,1))

J=icost(t0,X,y)
t=grad_des(t0,X,y,alpha,iters)
p1=10000*array([1,3.5]).dot(t)[0]
p2=10000*array([1,7]).dot(t)[0]
print(J,p1,p2)
print('MATLAB: 32.0727 4519.767868 45342.450129')
