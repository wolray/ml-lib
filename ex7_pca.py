from lib import *

data=io.loadmat('ex7data1.mat')
x=data['X']
x=NormFeat(x)

k=1

z=PCA(x,k)
Print4(z[0])
print('MATLAB: 1.4813')

x_re=PCABack(x,k)
Print4(x_re[0,:])
print('MATLAB: -1.0474 -1.0474')
