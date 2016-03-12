from lib_lg import *

X,y=load_data('ex2data1.txt')
lamb=0
n0=X.shape[1]-1
t0=zeros((n0+1,1))

J0=cost(t0,X,y,lamb)
t,J=cfmin(t0,X,y,lamb)
prob=sigmoid(array([1,45,85]).dot(t))
print(J0,J,prob)
print('MATLAB: 0.693 0.203 0.776')

# lamb=1
# t,J=cfmin(t0,X,y,lamb)
# print(J)
