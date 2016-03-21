# init
from numpy import *
from scipy import io, optimize as op, special as sp
from sklearn import svm, grid_search

def add_ones(X):
    m=X.shape[0]
    return c_[ones((m,1)),X]

def kmeans(X,c0,iters):
    m,n=X.shape
    k=c0.shape[0]
    def clust(X,c):
        J=zeros((m,k))
        for i in range(m):
            for j in range(k):
                J[i,j]=sum((X[i,:]-c[j,:])**2)
        return J.argmin(1)
    def newcent(X,idx,k):
        c=zeros((k,n))
        for i in range(k):
            p=where(idx==i)[0]
            c[i,:]=sum(X[p,:],0)/len(p)
        return c
    c=c0
    for i in range(iters):
        idx=clust(X,c)
        c=newcent(X,idx,k)
    return c,idx

def nn_cost(t,n,X,yy,lamb):
    m=X.shape[0]
    xout=nx(len(n)-1,t,n,X)
    J=sum(-yy*log(xout)-(1-yy)*log(1-xout))/m
    for i in range(1,len(n)):
        J+=sum(nt(i,t,n)[1:]**2)*lamb/(2*m)
    return J

def stat(X,c=0):
    mu=X.mean(0)
    if c:
        sigma=cov(X.T)
    else:
        sigma=diag(X.var(0))
    return mu,sigma

def gauss(X,mu,sigma):
    n=X.shape[1]
    p=((2*pi)**n*linalg.det(sigma))**(-1/2)*exp(-1/2*((X-mu).dot(linalg.pinv(sigma))*(X-mu)).sum(1))
    return p.reshape((-1,1))

def epsilon(yval,pval):
    f1_best=eps_best=0
    for eps in linspace(amin(pval),amax(pval),1000):
        tp=sum((yval==1)*(pval<eps))
        fp=sum((yval==0)*(pval<eps))
        fn=sum((yval==1)*(pval>=eps))
        prec=tp/(tp+fp)
        rec=tp/(tp+fn)
        f1=2*prec*rec/(prec+rec)
        if f1>f1_best:
            f1_best=f1
            eps_best=eps
    return eps_best,f1_best

def nn_grad(t,n,X,yy,lamb):
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

def cf_cost(xt,n,Y,R,lamb):
    X,t=nxt(xt,n)
    return sum(((hi(X,t)-Y)*R)**2)/2+(sum(X**2)+sum(t**2))*lamb/2

def cf_grad(xt,n,Y,R,lamb):
    X,t=nxt(xt,n)
    X_grad=(hi(X,t)-Y)*R.dot(t.T)+X*lamb
    t_grad=((hi(X,t)-Y)*R).T.dot(X)+t*lamb
    return append(X_grad,t_grad)

def cost(t,X,y,lamb):
    m=X.shape[0]
    return sum(-y*log(h(X,t))-(1-y)*log(1-h(X,t)))/m+sum(t[1:]**2)*lamb/(2*m)

def grad(t,X,y,lamb):
    m=X.shape[0]
    return X.T.dot(h(X,t)-y)/m+r_[t[:1]*0,t[1:]]*lamb/m

def predict(t,X,y):
    m=X.shape[0]
    xout=h(X,t)
    p=argmax(xout,1)+1
    num=m-count_nonzero(p-y.ravel())
    return (num*100)/m

def hi(X,t):
    return X.dot(t)

def nxt(xt,n):
    X=xt[:n[0]*n[1]].reshape((n[0],n[1]))
    t=xt[n[0]*n[1]:].reshape((n[1],n[2]))
    return X,t

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

def nn_predict(t,n,X,y):
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

def norm_features(X):
    m=X.shape[0]
    X_mean=ones((m,1)).dot(mean(X,0).reshape((1,-1)))
    X_std=ones((m,1)).dot(std(X,0,ddof=1).reshape((1,-1)))
    return (X-X_mean)/X_std

def pca(X,k):
    m=X.shape[0]
    U,S,V=linalg.svd(X.T.dot(X)/m)
    return X.dot(U[:,:k])

def pca_back(X,k):
    m=X.shape[0]
    U,S,V=linalg.svd(X.T.dot(X)/m)
    return X.dot(U[:,:k]).dot(U[:,:k].T)

def print1(lis):
    out=''
    for i in lis:
        out+=' %.1f' %i
    print('\nPYTHON:'+out)

def print2(lis):
    out=''
    for i in lis:
        out+=' %.2f' %i
    print('\nPYTHON:'+out)

def print3(lis):
    out=''
    for i in lis:
        out+=' %.3f' %i
    print('\nPYTHON:'+out)

def icost(t,X,y,lamb):
    m=X.shape[0]
    return sum((hi(X,t)-y)**2)/(2*m)+sum(t[1:]**2)*lamb/(2*m)

def igrad(t,X,y,lamb):
    m=X.shape[0]
    return X.T.dot(hi(X,t)-y)/m+r_[t[:1]*0,t[1:]]*lamb/m

def grad_des(t0,X,y,lamb,alpha,iters):
    t=t0
    for i in range(iters):
        t-=alpha*igrad(t,X,y,lamb)
    return t

def norm_eqn(X,y):
    t=linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return t

def print4(lis):
    out=''
    for i in lis:
        out+=' %.4f' %i
    print('\nPYTHON:'+out)

def printd(lis):
    out=''
    for i in lis:
        out+=' %d' %i
    print('\nPYTHON:'+out)

def randc(X,k):
    m=X.shape[0]
    return X[random.choice(m,k),:]

def ys(y,n):
    m=y.size
    yy=zeros((m,n[-1]))
    for i in range(m):
        for j in range(n[-1]):
            yy[i,j]=(j==y[i]-1)
    return yy
