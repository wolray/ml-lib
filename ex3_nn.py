from lib import *

data=io.loadmat('ex3data1.mat')
x=data['X']
x=AddOnes(x)
y=data['y']

data=io.loadmat('ex3weights.mat')
t1_trans=data['Theta1']
t2_trans=data['Theta2']
t=append(t1_trans.T,t2_trans.T)
n=[400,25,10]
lamb=0

p=NnPredict(t,n,x,y)
Print2([p])
print('MATLAB: 97.5%')
