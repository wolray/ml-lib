from imp import reload
import lib
reload(lib)
from lib import *

data=io.loadmat('ex3data1.mat')
x=data['X']
x=AddOnes(x)
y=data['y']

data=io.loadmat('ex4weights.mat')
t1_trans=data['Theta1']
t2_trans=data['Theta2']
t=append(t1_trans.T,t2_trans.T)
n=[400,25,10]
lamb=1
yy=Ny(y,n)

cost=NnCost(t,n,x,yy,lamb)
Print4([cost])
print('MATLAB: 0.3838')

t0=RandT(n)

t=op.fmin_cg(NnCost,fprime=NnGrad,x0=t0,args=(n,x,yy,lamb),maxiter=50,disp=False)
p=NnPredict(t,n,x,y)
Print2([p])
print('MATLAB: 95.3%')
