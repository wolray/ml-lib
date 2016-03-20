from lib_lg import *

data=io.loadmat('ex3data1.mat')
X=data['X']
X=add_ones(X)
y=data['y']

n=[X.shape[1]-1,10]
t0=zeros((n[0]+1,1))
lamb=0.1
tt=zeros((n[0]+1,n[1]))
yy=ys(n,y)

for k in range(n[-1]):
    tt[:,k]=cfmin_cg(t0,X,yy[:,k],lamb)
p=predict(tt,X,y)
print2([p])
print('MATLAB: 94.9%')
