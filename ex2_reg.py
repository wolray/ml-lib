from lib import *

X,y=load_data('ex2data1.txt')
lamb=[0,1,10,100]
n0=X.shape[1]-1
t0=zeros((n0+1,1))

for i in lamb:
    J0=ocost(t0,X,y,i)
    t,J=ofmin(t0,X,y,i)
    print('\nlamb=%d' %i)
    print(J0,J)
