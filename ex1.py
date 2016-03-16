from lib_ln import *

data=loadtxt('ex1data1.txt',delimiter=',')
X=data[:,:-1]
n0=X.shape[1]
X=add_ones(X)
y=data[:,-1:]

alpha=0.01
iters=1500
t0=zeros((n0+1,1))
lamb=0

J=cost(t0,X,y,lamb)
t=grad_des(t0,X,y,lamb,alpha,iters)
p1=10000*h(array([1,3.5]),t)[0]
p2=10000*h(array([1,7]),t)[0]

print4([J,p1,p2])
print('MATLAB: 32.0727 4519.7679 45342.4501')
