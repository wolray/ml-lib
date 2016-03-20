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

def ys(n,y):
    m=y.shape[0]
    yy=zeros((m,n[-1]))
    for i in range(m):
        for j in range(n[-1]):
            yy[i,j]=(j==y[i]-1)
    return yy
