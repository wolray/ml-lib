from lib_lg import *

data=loadtxt('ex2data1.txt',delimiter=',')
X=data[:,:-1]
n0=X.shape[1]
X=add_ones(X)
y=data[:,-1:]

lamb=0
t0=zeros((n0+1,1))

J0=cost(t0,X,y,lamb)
t,J=cfmin(t0,X,y,lamb)
prob=h(array([1,45,85]),t)

print3([J0,J,prob])
print('MATLAB: 0.693 0.203 0.776')
