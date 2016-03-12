# init
from lib import *

def h(X,t):
    return X.dot(t)

def cfmin(t0,X,y,lamb):
    y=y.flatten()
    result=op.fmin(cost,x0=t0,args=(X,y,lamb),maxiter=500,full_output=True)
    t,J=result[0],result[1]
    return t,J

def cost(t,X,y):
    m=y.shape[0]
    return sum((h(X,t)-y)**2)/(2*m)

def grad(t,X,y):
    m=y.shape[0]
    return X.T.dot(h(X,t)-y)/m

def grad_des(t0,X,y,alpha,iters):
    t=t0
    for i in range(iters):
        t=t-alpha*grad(t,X,y)
    return t

def feature_norm(X):
    m,n=X.shape
    X_mean=ones((m,1)).dot(mean(X,0).reshape((1,-1)))
    X_std=ones((m,1)).dot(std(X,0).reshape((1,-1)))
    return (X-X_mean)/X_std

def norm_eqn(X,y):
    t=linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return t
