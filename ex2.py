from lib import *

theta_0,X,y=load_data('ex2data1.txt')
lamb=0

theta,J=cfmin(theta_0,X,y,lamb)
prob=sigmoid(array([1,45,85]).dot(theta))
print(J,prob)
