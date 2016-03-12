# init
from lib import *

def cfmin(t0,X,y,lamb):
    y=y.flatten()
    result=op.fmin(cost,t0,args=(X,y,lamb),maxiter=500,full_output=True)
    t,J=result[0],result[1]
    return t,J

def cfmin_cg(t0,X,y,lamb):
    y=y.flatten()
    result=op.fmin_cg(cost,fprime=grad,x0=t0,args=(X,y,lamb),maxiter=50,disp=False,full_output=True)
    t=result[0]
    return t

def cost(t,X,y,lamb):
    m=X.shape[0]
    return sum(-y*log(h(X,t))-(1-y)*log(1-h(X,t)))/m+sum(t[1:]**2)*lamb/(2*m)

def grad(t,X,y,lamb):
    m=X.shape[0]
    return X.T.dot(h(X,t)-y)/m+sum(r_[0*t[:1],t[1:]*lamb/m])

def h(X,t):
    return sigmoid(X.dot(t))

def predict(t,X,y):
    m=X.shape[0]
    xout=h(X,t)
    p=argmax(X.dot(t),1)+1
    num=m-count_nonzero(p-y.flatten())
    print('%.2f%%' %(num*100/m))

def sigmoid(z):
    return sp.expit(z)
    # return 1/(1+exp(-z))
