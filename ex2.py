from numpy import *
from scipy import optimize as op

def sigmoid(z):
    return 1/(1+exp(-z))

def h0_log(X,theta):
    return sigmoid(X.dot(theta))

def costFunction(theta,X,y):
    m=X.shape[0]
    y=y.flatten()
    J=1/m*sum(-y*log(h0_log(X,theta))-(1-y)*log(1-h0_log(X,theta)))
    grad=1/m*h0_log(X,theta).T.dot(X)
    return J

def findmin(theta,X,y):
    result=op.fmin(costFunction,x0=theta,args=(X,y),maxiter=500,full_output=True)
    return result[0],result[1]

data=loadtxt('ex2data1.txt',delimiter=',')
m,n=data.shape
X=c_[ones((m,1)),data[:,:n-1]]
y=data[:,n-1:]
init_theta=zeros((n,1))

theta,J=findmin(init_theta,X,y)
prob=sigmoid(array([1,45,85]).dot(theta))
print(J,prob)
