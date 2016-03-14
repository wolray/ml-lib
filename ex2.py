from lib import *

X,y=load_data('ex2data1.txt')
lamb=0
n0=X.shape[1]-1
t0=zeros((n0+1,1))

J0=ocost(t0,X,y,lamb)
t,J=ofmin(t0,X,y,lamb)
prob=sigmoid(array([1,45,85]).dot(t))
print(J0,J,prob)
print('ex2: 0.693 0.203 0.776')
