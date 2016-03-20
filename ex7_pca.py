from lib import *

data=io.loadmat('ex7data1.mat')
X=data['X']
X=norm_features(X)

k=1

Z=pca(X,k)
print4(Z[0])
print('MATLAB: 1.4813')

X_re=pca_back(X,k)
print4(X_re[0,:])
print('MATLAB: -1.0474 -1.0474')
