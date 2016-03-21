from lib import *

data=io.loadmat('ex8_movies.mat')
Y=data['Y']
R=data['R']

data=io.loadmat('ex8_movieParams.mat')
X=data['X']
t=data['Theta'].T

n=[5,3,4]
X=X[:n[0],:n[1]]
t=t[:n[1],:n[2]]
xt0=append(X,t)
Y=Y[:n[0],:n[2]]
R=R[:n[0],:n[2]]

J=cf_cost(xt0,n,Y,R,0)
print4([J])
print('MATLAB: 22.2246')
