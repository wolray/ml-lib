from lib import *

theta_0,X,y=load_data('ex1data1.txt')

alpha=0.01
iters=1500

J=cost(theta_0,X,y)
theta=grad_descent(theta_0,X,y,alpha,iters)
p1=10000*array([1,3.5]).dot(theta)
p2=10000*array([1,7]).dot(theta)
myprint((J,theta,p1,p2))
