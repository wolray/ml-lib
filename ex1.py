from numpy import *

def computeCost(X,y,theta):
    m=size(y,0)
    J=1/(2*m)*sum((X*theta-y).A**2)
    return J

def gradientDescent(X,y,theta,alpha,num_iters):
    m=size(y,0)
    J_history=zeros((num_iters,1))
    for i in range(num_iters):
        theta=theta-alpha/m*X.T*(X*theta-y)
        J_history[i]=computeCost(X,y,theta)
        return [theta,J_history]

data=mat(loadtxt('ex1data1.txt',delimiter=','))
X=data[:,0]
y=data[:,1]
m=size(y,0)
X=c_[ones((m,1)),X]
theta=zeros((2,1))

iterations=1500
alpha=0.01

J=computeCost(X,y,theta)
theta=gradientDescent(X,y,theta,alpha,iterations)[0]
print(J)
