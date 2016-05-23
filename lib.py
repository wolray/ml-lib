# init
from numpy import *
from scipy import io, optimize as op, special as sp
from sklearn import svm, grid_search

def AddOnes(x):
    m=x.shape[0]
    return c_[ones((m,1)),x]

def CfPara(xt,n):
    x=xt[:n[0]*n[1]].reshape((n[0],n[1]))
    t=xt[n[0]*n[1]:].reshape((n[1],n[2]))
    return x,t

def CfCost(xt,n,y,r,lamb):
    x,t=CfPara(xt,n)
    return sum(((HLin(x,t)-y)*r)**2)/2+(sum(x**2)+sum(t**2))*lamb/2

def CfGrad(xt,n,y,r,lamb):
    x,t=CfPara(xt,n)
    x_grad=(HLin(x,t)-y)*r.dot(t.T)+x*lamb
    t_grad=((HLin(x,t)-y)*r).T.dot(x)+t*lamb
    return append(x_grad,t_grad)

def Cost(t,x,y,lamb):
    m=x.shape[0]
    return sum(-y*log(H(x,t))-(1-y)*log(1-H(x,t)))/m+sum(t[1:]**2)*lamb/(2*m)

def Epsilon(yval,pval):
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

def Gauss(x,mu,sigma):
    n=x.shape[1]
    p=((2*pi)**n*linalg.det(sigma))**(-1/2)*exp(-1/2*((x-mu).dot(linalg.pinv(sigma))*(x-mu)).sum(1))
    return p.reshape((-1,1))

def Grad(t,x,y,lamb):
    m=x.shape[0]
    return x.T.dot(H(x,t)-y)/m+r_[t[:1]*0,t[1:]]*lamb/m

def GradDes(t0,x,y,lamb,alpha,iters):
    t=t0
    for i in range(iters):
        t-=alpha*LinGrad(t,x,y,lamb)
    return t

def H(x,t):
    return sp.expit(x.dot(t)) # 1/(1+exp(-z))

def HG(x):
    return x*(1-x)

def HLin(x,t):
    return x.dot(t)

def LinCost(t,x,y,lamb):
    m=x.shape[0]
    return sum((HLin(x,t)-y)**2)/(2*m)+sum(t[1:]**2)*lamb/(2*m)

def LinGrad(t,x,y,lamb):
    m=x.shape[0]
    return x.T.dot(HLin(x,t)-y)/m+r_[t[:1]*0,t[1:]]*lamb/m

def Kmeans(x,c0,iters):
    m,n=x.shape
    k=c0.shape[0]
    def Clust(x,c):
        cost=zeros((m,k))
        for i in range(m):
            for j in range(k):
                cost[i,j]=sum((x[i,:]-c[j,:])**2)
        return cost.argmin(1)
    def NewCent(x,idx,k):
        c=zeros((k,n))
        for i in range(k):
            p=where(idx==i)[0]
            c[i,:]=x[p,:].sum(0)/len(p)
        return c
    c=c0
    for i in range(iters):
        idx=Clust(x,c)
        c=NewCent(x,idx,k)
    return c,idx

def NnCost(t,n,x,yy,lamb):
    m=x.shape[0]
    xout=Nx(len(n)-1,t,n,x)
    cost=sum(-yy*log(xout)-(1-yy)*log(1-xout))/m
    for i in range(1,len(n)):
        cost+=sum(Nt(i,t,n)[1:]**2)*lamb/(2*m)
    return cost

def NnGrad(t,n,x,yy,lamb):
    def Dt(k):
        if 0<k<len(n)-1:
            return Dt(k+1).dot(Nt(k+1,t,n)[1:].T)*HG(Nx(k,t,n,x)[:,1:])
        elif k==len(n)-1:
            return Nx(k,t,n,x)-yy
    def Gg(k):
        m=x.shape[0]
        return Nx(k-1,t,n,x).T.dot(Dt(k))/m+r_[Nt(k,t,n)[:1]*0,Nt(k,t,n)[1:]*lamb/m]
    g=[]
    for i in range(1,len(n)):
        g=append(g,Gg(i))
    return g

def NnPredict(t,n,x,y):
    m=x.shape[0]
    xout=Nx(len(n)-1,t,n,x)
    p=xout.argmax(1)+1
    num=m-count_nonzero(p-y.ravel())
    return num*100/m

def NormEqn(x,y):
    t=linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return t

def NormFeat(x):
    m=x.shape[0]
    x_mean=ones((m,1)).dot(mean(x,0).reshape((1,-1)))
    x_std=ones((m,1)).dot(std(x,0,ddof=1).reshape((1,-1)))
    return (x-x_mean)/x_std

def Nt(k,t,n):
    def Nk(k):
        return (n[k-1]+1)*n[k]
    if k==1:
        return t[:Nk(k)].reshape(((n[k-1]+1),n[k]))
    elif 1<k<len(n):
        return t[Nk(k-1):Nk(k-1)+Nk(k)].reshape(((n[k-1]+1),n[k]))

def Nx(k,t,n,x):
    if k==0:
        return x
    elif 0<k<len(n)-1:
        return AddOnes(H(Nx(k-1,t,n,x),Nt(k,t,n)))
    elif k==len(n)-1:
        return H(Nx(k-1,t,n,x),Nt(k,t,n))

def PCA(x,k):
    m=x.shape[0]
    uu,ss,vv=linalg.svd(x.T.dot(x)/m)
    return x.dot(uu[:,:k])

def PCABack(x,k):
    m=x.shape[0]
    uu,ss,vv=linalg.svd(x.T.dot(x)/m)
    return x.dot(uu[:,:k]).dot(uu[:,:k].T)

def Predict(t,x,y):
    m=x.shape[0]
    xout=H(x,t)
    p=argmax(xout,1)+1
    num=m-count_nonzero(p-y)
    return (num*100)/m

def Print1(lis):
    out=''
    for i in lis:
        out+=' %.1f' %i
    print('\nPython:'+out)

def Print2(lis):
    out=''
    for i in lis:
        out+=' %.2f' %i
    print('\nPython:'+out)

def Print3(lis):
    out=''
    for i in lis:
        out+=' %.3f' %i
    print('\nPython:'+out)

def Print4(lis):
    out=''
    for i in lis:
        out+=' %.4f' %i
    print('\nPython:'+out)

def Printd(lis):
    out=''
    for i in lis:
        out+=' %d' %i
    print('\nPython:'+out)

def RandC(x,k):
    m=x.shape[0]
    return x[random.choice(m,k),:]

def RandT(n):
    t_count=0
    for i in range(len(n)-1):
        t_count+=(n[i]+1)*n[i+1]
    return 1-2*random.random(t_count)

def Stat(x,c=0):
    mu=x.mean(0)
    if c:
        sigma=cov(x.T)
    else:
        sigma=diag(x.var(0))
    return mu,sigma

def Ny(y,n):
    m=y.size
    yy=zeros((m,n[-1]))
    for i in range(m):
        for j in range(n[-1]):
            yy[i,j]=(j==y[i]-1)
    return yy
