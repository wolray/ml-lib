from numpy import *

data=mat(loadtxt('ex1data1.txt',delimiter=','))
X=data[:,0]
y=data[:,1]
m=size(y,0)
X=column_stack([ones((m,1)),data[:,0]])
theta=zeros((2,1))

iterations=1500
alpha=0.01

from ex1def import *
J=computeCost(X,y,theta)
theta=gradientDescent(X,y,theta,alpha,iterations)[0]
