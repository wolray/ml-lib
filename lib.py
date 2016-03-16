# init
from numpy import *
from scipy import io, optimize as op, special as sp
from sklearn import svm, grid_search

def add_ones(X):
    m=X.shape[0]
    return c_[ones((m,1)),X]

def norm_features(X):
    m=X.shape[0]
    X_mean=ones((m,1)).dot(mean(X,0).reshape((1,-1)))
    X_std=ones((m,1)).dot(std(X,0).reshape((1,-1)))
    X_norm=(X-X_mean)/X_std
    return add_ones(X_norm)

def print1(lis):
    out=''
    for i in lis:
        out+='%.1f ' %i
    print('\nPYTHON: '+out)

def print2(lis):
    out=''
    for i in lis:
        out+='%.2f ' %i
    print('\nPYTHON: '+out)

def print3(lis):
    out=''
    for i in lis:
        out+='%.3f ' %i
    print('\nPYTHON: '+out)

def print4(lis):
    out=''
    for i in lis:
        out+='%.4f ' %i
    print('\nPYTHON: '+out)

def printd(lis):
    out=''
    for i in lis:
        out+='%d ' %i
    print('\nPYTHON: '+out)

def ys(n,y):
    m=y.shape[0]
    yy=zeros((m,n[-1]))
    for i in range(m):
        for j in range(n[-1]):
            yy[i,j]=(j==y[i]-1)
    return yy
