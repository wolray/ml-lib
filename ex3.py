from imp import reload
import lib
reload(lib)
from lib import *

data=io.loadmat('ex3data1.mat')
x=data['X']
x=AddOnes(x)
y=data['y'].ravel()

n=[x.shape[1]-1,10]
t0=zeros(n[0]+1)
lamb=0.1
tt=zeros((n[0]+1,n[1]))
yy=Ny(y,n)

for k in range(n[-1]):
    tt[:,k]=op.fmin_cg(Cost,fprime=Grad,x0=t0,args=(x,yy[:,k],lamb),maxiter=50,disp=False)
p=Predict(tt,x,y)
Print2([p])
print('MATLAB: 94.9%')
