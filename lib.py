# init
from numpy import *
from scipy import io, optimize as op, special as sp

def grad_des(t0,X,y,alpha,iters):
    for i in range(iters):
        t0-=alpha*igrad(t0,X,y)
    return t0

def h(X,t):
    return sigmoid(X.dot(t))

def hg(x):
    return x*(1-x)

def hi(X,t):
    return X.dot(t)

def icost(t,X,y):
    m=y.shape[0]
    return sum((hi(X,t)-y)**2)/(2*m)

def ifmin(t0,X,y):
    y=y.flatten()
    result=op.fmin(icost,t0,args=(X,y),maxiter=500,disp=False,full_output=True)
    t,J=result[0],result[1]
    return t,J

def igrad(t,X,y):
    m=y.shape[0]
    return X.T.dot(hi(X,t)-y)/m

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

def ncost(t,n,X,yk,lamb):
    m=X.shape[0]
    xout=xx(len(n)-1,t,n,X)
    J=sum(-yk*log(xout)-(1-yk)*log(1-xout))/m
    for k in range(1,len(n)):
        J+=sum(tt(k,t,n)[1:]**2)*lamb/(2*m)
    return J

def nfmin_cg(t0,n,X,yk,lamb):
    return op.fmin_cg(ncost,fprime=ngrad,x0=t0,args=(n,X,yk,lamb),maxiter=50,disp=False)

def ngrad(t,n,X,yk,lamb):
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

def norm_eqn(X,y):
    t=linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return t

def norm_feature(X):
    m=X.shape[0]
    X=X[:,1:]
    X_mean=ones((m,1)).dot(mean(X,0).reshape((1,-1)))
    X_std=ones((m,1)).dot(std(X,0).reshape((1,-1)))
    X_norm=(X-X_mean)/X_std
    return c_[ones((m,1)),X_norm]

def npredict(t,n,X,y):
    m=X.shape[0]
    xout=xx(len(n)-1,t,n,X)
    p=argmax(xout,1)+1
    num=m-count_nonzero(p-y.flatten())
    print('%.2f%%' %(num*100/m))

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
    return X.T.dot(h(X,t)-y)/m+sum(r_[0*t[:1],t[1:]*lamb/m])

def opredict(t,X,y):
    m=X.shape[0]
    xout=h(X,t)
    p=argmax(X.dot(t),1)+1
    num=m-count_nonzero(p-y.flatten())
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
