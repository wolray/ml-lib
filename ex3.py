from lib import *

data=io.loadmat('ex3data1.mat')
X=data['X']
X=add_ones(X)
y=data['y']

n=[X.shape[1]-1,10]
t0=zeros(n[0]+1)
lamb=0.1
tt=zeros((n[0]+1,n[1]))
yy=ys(y,n)

for k in range(n[-1]):
    tt[:,k]=op.fmin_cg(cost,fprime=grad,x0=t0,args=(X,yy[:,k],lamb),maxiter=50,disp=False)
p=predict(tt,X,y)
print2([p])
print('MATLAB: 94.9%')
