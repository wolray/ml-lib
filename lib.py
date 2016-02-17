from numpy import *
from scipy import optimize as op

def sigmoid(z):
    return 1/(1+exp(-z))

def h0(X,theta):
    return X.dot(theta)

def h0_log(X,theta):
    return sigmoid(X.dot(theta))

def grad(theta,X,y):
    return X.T.dot(h0(X,theta)-y)/m

def grad_reg(theta,X,y,lamb):
    m=y.shape[0]
    grad=X.T.dot(h0_log(X,theta)-y)/m
    grad[1:]=grad[1:]+t[1:]*lamb/m
    return grad

def grad_descent(theta,X,y,alpha,num_iters):
    m=y.shape[0]
    for i in range(num_iters):
        theta=theta-alpha*grad
    return theta

def cost(theta,X,y):
    m=y.shape[0]
    J=sum((h0(X,theta)-y)**2)/(2*m)
    return J

def cost_log(theta,X,y):
    m=y.shape[0]
    y=y.flatten()
    term1=-y*log(h0_log(X,theta))
    term2=-(1-y)*log(1-h0_log(X,theta))
    J=sum(term1+term2)/m
    return J

def cfmin(theta,X,y):
    result=op.fmin(cost_log,x0=theta,args=(X,y),maxiter=500,full_output=True)
    return result[0],result[1]

def load_data(filename):
    data=loadtxt(filename,delimiter=',')
    m,n=data.shape
    X=c_[ones((m,1)),data[:,:n-1]]
    y=data[:,n-1:]
    theta0=zeros((n,1))
    return theta0,X,y
