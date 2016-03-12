# init
from numpy import *
from scipy import io, optimize as op, special as sp

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
