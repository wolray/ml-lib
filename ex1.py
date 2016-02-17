from lib import *

theta0,X,y=load_data('ex1data1.txt')

alpha=0.01
iters=1500

J=cost(theta0,X,y)
theta=grad_descent(theta0,X,y,alpha,iters)
print(J)
