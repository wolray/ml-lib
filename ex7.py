from imp import reload
import lib
reload(lib)
from lib import *

data=io.loadmat('ex7data2.mat')
x=data['X']

c0=array([[3,3],[6,2],[8,5]])
iters=10

c,idx=Kmeans(x,c0,iters)
Print4(c.ravel())
print('MATLAB: 1.9540 5.0256 3.0437 1.0154 6.0337 3.0005')
