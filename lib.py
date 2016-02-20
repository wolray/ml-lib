from numpy import *
from scipy import io, optimize as op, special as sp

def myprint(t):
    print()
    for i in t:
        print(i.flatten())

def feature_norm(X):
    m,n=X.shape
    X_mean=ones((m,1)).dot(mean(X,0).reshape((1,-1)))
    X_std=ones((m,1)).dot(std(X,0).reshape((1,-1)))
    return (X-X_mean)/X_std

def load_data(filename,*norm):
    data=loadtxt(filename,delimiter=',')
    X=data[:,:-1]
    y=data[:,-1:]
    m,n=data.shape
    if norm and norm[0]!=0:
        X=feature_norm(X)
    X=c_[ones((m,1)),data[:,:-1]]
    theta_0=zeros((n,1))
    return theta_0,X,y

def load_mat(filename,*norm):
    data=io.loadmat(filename)
    X=data['X']
    y=data['y']
    m,n=X.shape[0],X.shape[1]+1
    if norm and norm[0]!=0:
        X=feature_norm(X)
    X=c_[ones((m,1)),data['X']]
    theta_0=zeros((n,1))
    return theta_0,X,y

def sigmoid(z):
    return sp.expit(z)
    # return 1/(1+exp(-z))

def h0(X,theta):
    return X.dot(theta)

def h0_log(X,theta):
    return sigmoid(X.dot(theta))

def grad(theta,X,y):
    m=y.shape[0]
    return X.T.dot(h0(X,theta)-y)/m

def grad_log(theta,X,y,lamb):
    m=y.shape[0]
    g=X.T.dot(h0_log(X,theta)-y)/m
    g[1:]=g[1:]+theta[1:]*lamb/m
    return g

def grad_descent(theta_0,X,y,alpha,iters):
    theta=theta_0
    for i in range(iters):
        theta=theta-alpha*grad(theta,X,y)
    return theta

def norm_eqn(X,y):
    theta=linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

def cost(theta,X,y):
    m=y.shape[0]
    return sum((h0(X,theta)-y)**2)/(2*m)

def cost_log(theta,X,y,lamb):
    m=y.shape[0]
    term1=-y*log(h0_log(X,theta))
    term2=-(1-y)*log(1-h0_log(X,theta))
    return sum(term1+term2)/m+sum(theta**2)*lamb/(2*m)

def cfmin(theta_0,X,y,lamb):
    y=y.flatten()
    result=op.fmin(cost_log,x0=theta_0,args=(X,y,lamb),maxiter=500,full_output=True)
    theta,J=result[0],result[1]
    return theta,J

def cfmin_cg(theta_0,X,y,lamb):
    y=y.flatten()
    result=op.fmin_cg(cost_log,fprime=grad_log,x0=theta_0,args=(X,y,lamb),maxiter=50,disp=False,full_output=True)
    theta=result[0]
    return theta

def one_all(theta_0,X,y,num_labels,lamb):
    n=X.shape[1]
    theta_all=zeros((n,num_labels))
    for k in range(num_labels):
        yk=((y==k+1)+0)
        theta_all[:,k]=cfmin_cg(theta_0,X,yk,lamb)
    return theta_all

def predict_one_all(theta_all,X,y):
    m=y.shape[0]
    count=0
    for i in range(m):
        p=argmax(X[i].dot(theta_all))+1
        if p==y[i]:
            count+=1
    print('accuracy: %.2f%%' % (count*100/m))
