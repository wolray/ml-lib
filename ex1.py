from imp import reload
import lib
reload(lib)
from lib import *

data=loadtxt('ex1data1.txt',delimiter=',')
x=data[:,:-1]
n0=x.shape[1]
x=AddOnes(x)
y=data[:,-1:].ravel()

alpha=0.01
iters=1500
t0=zeros(n0+1)
lamb=0

cost=LinCost(t0,x,y,lamb)
t=GradDes(t0,x,y,lamb,alpha,iters)
p1=10000*HLin(array([1,3.5]),t)
p2=10000*HLin(array([1,7]),t)
Print4([cost,p1,p2])
print('MATLAB: 32.0727 4519.7679 45342.4501')
