from lib import *

data=io.loadmat('ex8data1.mat')
x=data['X']
xval=data['Xval']
yval=data['yval']

mu,sigma=Stat(x)
pval=Gauss(xval,mu,sigma)
eps,f1=Epsilon(yval,pval)
print('\nPYTHON: '+str(eps))
print('MATLAB: 8.99e-05')

data=io.loadmat('ex8data2.mat')
x=data['X']
xval=data['Xval']
yval=data['yval']

mu,sigma=Stat(x)
pval=Gauss(xval,mu,sigma)
eps,f1=Epsilon(yval,pval)
print('\nPYTHON: '+str(eps))
print('MATLAB: 1.38e-18')
