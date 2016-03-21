from lib import *

data=io.loadmat('ex3data1.mat')
X=data['X']
X=add_ones(X)
y=data['y']

data=io.loadmat('ex4weights.mat')
t1_trans=data['Theta1']
t2_trans=data['Theta2']
t=append(t1_trans.T,t2_trans.T)
n=[400,25,10]
lamb=1
yy=ys(y,n)

J=nn_cost(t,n,X,yy,lamb)
print4([J])
print('MATLAB: 0.3838')

t0=randt(n)

t=op.fmin_cg(nn_cost,fprime=nn_grad,x0=t0,args=(n,X,yy,lamb),maxiter=50,disp=False)
p=nn_predict(t,n,X,y)
print2([p])
print('MATLAB: 95.3%')
