from lib import *

X,y=load_mat('ex3data1.mat')
n=[X.shape[1]-1,10]
t0=zeros((n[0]+1,1))

lamb=0.1
tt=zeros((n[0]+1,n[1]))

for k in range(n[-1]):
    tt[:,k]=ofmin_cg(t0,X,(y==k+1)+0,lamb)

opredict(tt,X,y)
print('ex3: 94.9%')
