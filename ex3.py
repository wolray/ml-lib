from lib_lg import *

X,y=load_mat('ex3data1.mat')
n=[X.shape[1]-1,10]
t0=zeros((n[0]+1,1))

lamb=0.1
tk=zeros((n[0]+1,n[1]))

for k in range(n[-1]):
    tk[:,k]=cfmin_cg(t0,X,(y==k+1)+0,lamb)

predict(tk,n,X,y)
print('pdf: 94.9%')
