from imp import reload
import lib
reload(lib)
from lib import *

data=loadtxt('ex2data1.txt',delimiter=',')
x=data[:,:-1]
x=AddOnes(x)
y=data[:,-1:].ravel()

lamb=0
t0=zeros(x.shape[1])

cost0=Cost(t0,x,y,lamb)
out=op.fmin(Cost,t0,args=(x,y,lamb),maxiter=500,disp=False,full_output=True)
t,cost=out[0],out[1]
prob=H(array([1,45,85]),t)
Print3([cost0,cost,prob])
print('MATLAB: 0.693 0.203 0.776')
