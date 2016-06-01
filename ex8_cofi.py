from imp import reload
import lib
reload(lib)
from lib import *

data=io.loadmat('ex8_movies.mat')
y=data['Y']
r=data['R']

data=io.loadmat('ex8_movieParams.mat')
x=data['X']
t=data['Theta'].T

n=[5,3,4]
x=x[:n[0],:n[1]]
t=t[:n[1],:n[2]]
xt0=append(x,t)
y=y[:n[0],:n[2]]
r=r[:n[0],:n[2]]

cost=CfCost(xt0,n,y,r,0)
Print4([cost])
print('MATLAB: 22.2246')
