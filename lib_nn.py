# init
from lib import *

def cfmin_cg(t0,n,X,yy,lamb):
    return op.fmin_cg(cost,fprime=grad,x0=t0,args=(n,X,yy,lamb),maxiter=50,disp=False)

def cost(t,n,X,yy,lamb):
    m=X.shape[0]
    xout=nx(len(n)-1,t,n,X)
    J=sum(-yy*log(xout)-(1-yy)*log(1-xout))/m
    for i in range(1,len(n)):
        J+=sum(nt(i,t,n)[1:]**2)*lamb/(2*m)
    return J

def grad(t,n,X,yy,lamb):
    def dt(k):
        if 0<k<len(n)-1:
            return dt(k+1).dot(nt(k+1,t,n)[1:].T)*hg(nx(k,t,n,X)[:,1:])
        elif k==len(n)-1:
            return nx(k,t,n,X)-yy
    def gg(k):
        m=X.shape[0]
        return nx(k-1,t,n,X).T.dot(dt(k))/m+r_[nt(k,t,n)[:1]*0,nt(k,t,n)[1:]*lamb/m]
    g=[]
    for i in range(1,len(n)):
        g=append(g,gg(i))
    return g

def h(X,t):
    return sp.expit(X.dot(t)) # 1/(1+exp(-z))

def hg(x):
    return x*(1-x)

def nt(k,t,n):
    def nk(k):
        return (n[k-1]+1)*n[k]
    if k==1:
        return t[:nk(k)].reshape(((n[k-1]+1),n[k]))
    elif 1<k<len(n):
        return t[nk(k-1):nk(k-1)+nk(k)].reshape(((n[k-1]+1),n[k]))

def nx(k,t,n,X):
    if k==0:
        return X
    elif 0<k<len(n)-1:
        return add_ones(h(nx(k-1,t,n,X),nt(k,t,n)))
    elif k==len(n)-1:
        return h(nx(k-1,t,n,X),nt(k,t,n))

def predict(t,n,X,y):
    m=X.shape[0]
    xout=nx(len(n)-1,t,n,X)
    p=xout.argmax(1)+1
    num=m-count_nonzero(p-y.ravel())
    return num*100/m

def randt(n):
    t_count=0
    for i in range(len(n)-1):
        t_count+=(n[i]+1)*n[i+1]
    return 1-2*random.random(t_count)
