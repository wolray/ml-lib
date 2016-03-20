# init
from lib import *

def cfmin(t0,X,y,lamb):
    y=y.ravel()
    result=op.fmin(cost,t0,args=(X,y,lamb),maxiter=500,disp=False,full_output=True)
    t,J=result[0],result[1]
    return t,J

def cfmin_cg(t0,X,y,lamb):
    y=y.ravel()
    return op.fmin_cg(cost,fprime=grad,x0=t0,args=(X,y,lamb),maxiter=50,disp=False)

def cost(t,X,y,lamb):
    m=X.shape[0]
    return sum((h(X,t)-y)**2)/(2*m)+sum(t[1:]**2)*lamb/(2*m)

def grad(t,X,y,lamb):
    m=X.shape[0]
    return X.T.dot(h(X,t)-y)/m+r_[t[:1]*0,t[1:]]*lamb/m

def grad_des(t0,X,y,lamb,alpha,iters):
    t=t0
    for i in range(iters):
        t-=alpha*grad(t,X,y,lamb)
    return t

def h(X,t):
    return X.dot(t)

def norm_eqn(X,y):
    t=linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return t
