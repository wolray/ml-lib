# init
from lib import *

def cfmin_cg(t0,n,X,yk,lamb):
    result=op.fmin_cg(cost,fprime=grad,x0=t0,args=(n,X,yk,lamb),maxiter=50,disp=False,full_output=True)
    t=result[0]
    return t

def cost(t,n,X,yk,lamb):
    m=X.shape[0]
    xout=xx(len(n)-1,t,n,X)
    J=sum(-yk*log(xout)-(1-yk)*log(1-xout))/m
    for k in range(1,len(n)):
        J+=sum(tt(k,t,n)[1:]**2)*lamb/(2*m)
    return J

def grad(t,n,X,yk,lamb):
    def dt(k):
        if 0<k<len(n)-1:
            return dt(k+1).dot(tt(k+1,t,n)[1:].T)*hg(xx(k,t,n,X)[:,1:])
        elif k==len(n)-1:
            return xx(k,t,n,X)-yk
    def gg(k):
        m=X.shape[0]
        return xx(k-1,t,n,X).T.dot(dt(k))/m+r_[0*tt(k,t,n)[:1],tt(k,t,n)[1:]*lamb/m]
    g=[]
    for k in range(1,len(n)):
        g=append(g,gg(k))
    return g

def h(X,t):
    return sigmoid(X.dot(t))

def hg(x):
    return x*(1-x)

def predict(t,n,X,y):
    m=X.shape[0]
    xout=xx(len(n)-1,t,n,X)
    p=argmax(xout,1)+1
    num=m-count_nonzero(p.reshape(-1,1)-y)
    print('%.2f%%' %(num*100/m))

def sigmoid(z):
    return sp.expit(z)
    # return 1/(1+exp(-z))

def tt(k,t,n):
    def nk(k):
        return (n[k-1]+1)*n[k]
    if k==1:
        return t[:nk(k)].reshape(((n[k-1]+1),n[k]))
    elif 1<k<len(n):
        return t[nk(k-1):nk(k-1)+nk(k)].reshape(((n[k-1]+1),n[k]))

def xx(k,t,n,X):
    m=X.shape[0]
    if k==0:
        return X
    elif 0<k<len(n)-1:
        return c_[ones((m,1)),h(xx(k-1,t,n,X),tt(k,t,n))]
    elif k==len(n)-1:
        return h(xx(k-1,t,n,X),tt(k,t,n))

def yy(n,y):
    m=y.shape[0]
    yk=zeros((m,n[-1]))
    for i in range(m):
        for j in range(n[-1]):
            yk[i,j]=(j==y[i]-1)
    return yk
