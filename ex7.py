from lib import *

data=io.loadmat('ex7data2.mat')
X=data['X']

c0=array([[3,3],[6,2],[8,5]])
iters=10

c,idx=kmeans(X,c0,iters)
print4(c.ravel())
print('MATLAB: 1.9540 5.0256 3.0437 1.0154 6.0337 3.0005')
