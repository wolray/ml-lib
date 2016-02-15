from numpy import *

def h0(X,theta):
    return X.dot(theta)

def computeCost(X,y,theta):
    m=y.shape[0]
    J=1/(2*m)*sum((h0(X,theta)-y)**2)
    return J

def gradientDescent(X,y,theta,alpha,num_iters):
    m=y.shape[0]
    J_history=zeros((num_iters,1))
    for i in range(num_iters):
        theta=theta-alpha/m*X.T.dot(h0(X,theta)-y)
        J_history[i]=computeCost(X,y,theta)
        return theta,J_history

data=loadtxt('ex1data1.txt',delimiter=',')
m,n=data.shape
X=c_[ones((m,1)),data[:,:n-1]]
y=data[:,n-1:]
init_theta=zeros((n,1))

iterations=1500
alpha=0.01

J=computeCost(X,y,init_theta)
theta=gradientDescent(X,y,init_theta,alpha,iterations)[0]
print(J)
