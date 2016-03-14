# init
from numpy import *
from scipy import io, optimize as op, special as sp

def h(X,t):
    return sp.expit(X.dot(t)) # 1/(1+exp(-z))

def hg(x):
    return x*(1-x)

def hi(X,t):
    return X.dot(t)

def icost(t,X,y,lamb):
    m=y.shape[0]
    return sum((hi(X,t)-y)**2)/(2*m)+sum(t[1:]**2)*lamb/(2*m)

def ifmin(t0,X,y,lamb):
    y=y.flatten()
    result=op.fmin(icost,t0,args=(X,y,lamb),maxiter=500,disp=False,full_output=True)
    t,J=result[0],result[1]
    return t,J

def ifmin_cg(t0,X,y,lamb):
    y=y.flatten()
    return op.fmin_cg(icost,fprime=igrad,x0=t0,args=(X,y,lamb),maxiter=50,disp=False)

def igrad(t,X,y,lamb):
    m=X.shape[0]
    return X.T.dot(hi(X,t)-y)/m+r_[t[:1]*0,t[1:]]*lamb/m

def igrad_des(t0,X,y,lamb,alpha,iters):
    for i in range(iters):
        t0-=alpha*igrad(t0,X,y,lamb)
    return t0

def inorm_eqn(X,y):
    t=linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return t

def inorm_feature(X):
    m=X.shape[0]
    X=X[:,1:]
    X_mean=ones((m,1)).dot(mean(X,0).reshape((1,-1)))
    X_std=ones((m,1)).dot(std(X,0).reshape((1,-1)))
    X_norm=(X-X_mean)/X_std
    return c_[ones((m,1)),X_norm]

def load_data(filename):
    data=loadtxt(filename,delimiter=',')
    X=data[:,:-1]
    y=data[:,-1:]
    m=X.shape[0]
    X=c_[ones((m,1)),X]
    return X,y

def load_mat(filename):
    data=io.loadmat(filename)
    X=data['X']
    y=data['y']
    m=X.shape[0]
    X=c_[ones((m,1)),X]
    return X,y

def load_nn(filename):
    data=io.loadmat(filename)
    t1=data['Theta1']
    t2=data['Theta2']
    return t1,t2

def ncost(t,n,X,yy,lamb):
    m=X.shape[0]
    xout=nx(len(n)-1,t,n,X)
    J=sum(-yy*log(xout)-(1-yy)*log(1-xout))/m
    for i in range(1,len(n)):
        J+=sum(nt(i,t,n)[1:]**2)*lamb/(2*m)
    return J

def nfmin_cg(t0,n,X,yy,lamb):
    return op.fmin_cg(ncost,fprime=ngrad,x0=t0,args=(n,X,yy,lamb),maxiter=50,disp=False)

def ngrad(t,n,X,yy,lamb):
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

def npredict(t,n,X,y):
    m=X.shape[0]
    xout=nx(len(n)-1,t,n,X)
    p=argmax(xout,1)+1
    num=m-count_nonzero(p-y.flatten())
    print('%.2f%%' %(num*100/m))

def nrandt(n):
    t_count=0
    for i in range(len(n)-1):
        t_count+=(n[i]+1)*n[i+1]
    return 1-2*random.random(t_count)

def nt(k,t,n):
    def nk(k):
        return (n[k-1]+1)*n[k]
    if k==1:
        return t[:nk(k)].reshape(((n[k-1]+1),n[k]))
    elif 1<k<len(n):
        return t[nk(k-1):nk(k-1)+nk(k)].reshape(((n[k-1]+1),n[k]))

def nx(k,t,n,X):
    m=X.shape[0]
    if k==0:
        return X
    elif 0<k<len(n)-1:
        return c_[ones((m,1)),h(nx(k-1,t,n,X),nt(k,t,n))]
    elif k==len(n)-1:
        return h(nx(k-1,t,n,X),nt(k,t,n))

def nyy(n,y):
    m=y.shape[0]
    yy=zeros((m,n[-1]))
    for i in range(m):
        for j in range(n[-1]):
            yy[i,j]=(j==y[i]-1)
    return yy

def ocost(t,X,y,lamb):
    m=X.shape[0]
    return sum(-y*log(h(X,t))-(1-y)*log(1-h(X,t)))/m+sum(t[1:]**2)*lamb/(2*m)

def ofmin(t0,X,y,lamb):
    y=y.flatten()
    result=op.fmin(ocost,t0,args=(X,y,lamb),maxiter=500,disp=False,full_output=True)
    t,J=result[0],result[1]
    return t,J

def ofmin_cg(t0,X,y,lamb):
    y=y.flatten()
    return op.fmin_cg(ocost,fprime=ograd,x0=t0,args=(X,y,lamb),maxiter=50,disp=False)

def ograd(t,X,y,lamb):
    m=X.shape[0]
    return X.T.dot(h(X,t)-y)/m+r_[t[:1]*0,t[1:]*lamb/m]

def opredict(t,X,y):
    m=X.shape[0]
    xout=h(X,t)
    p=argmax(X.dot(t),1)+1
    num=m-count_nonzero(p-y.flatten())
    print('%.2f%%' %(num*100/m))
