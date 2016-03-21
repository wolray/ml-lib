from lib import *

data=io.loadmat('ex8data1.mat')
X=data['X']
Xval=data['Xval']
yval=data['yval']

mu,sigma=stat(X)
pval=gauss(Xval,mu,sigma)
eps,f1=epsilon(yval,pval)
print('\nPYTHON: '+str(eps))
print('MATLAB: 8.99e-05')

data=io.loadmat('ex8data2.mat')
X=data['X']
Xval=data['Xval']
yval=data['yval']

mu,sigma=stat(X)
pval=gauss(Xval,mu,sigma)
eps,f1=epsilon(yval,pval)
print('\nPYTHON: '+str(eps))
print('MATLAB: 1.38e-18')
