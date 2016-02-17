from lib import *

theta0,X,y=load_data('ex2data1.txt')

theta,J=cfmin(theta0,X,y)
prob=sigmoid(array([1,45,85]).dot(theta))
print(J,prob)
