from lib import *

data=io.loadmat('ex6data3.mat')
x=data['X']
y=data['y'].ravel()
xval=data['Xval']
yval=data['yval'].ravel()

clf=svm.SVC(C=1,gamma=0.01)
svm1=clf.fit(x,y)
p1=svm1.predict(xval)
Printd(p1[:20])
print('MATLAB: 1 1 0 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1')

cs=sigmas=array([0.01,0.03,0.1,0.3,1,3,10,30])
gammas=1/(sigmas**2)
paras={'C':cs,'gamma':gammas}
grid=grid_search.GridSearchCV(clf,paras)
svm2=grid.fit(x,y)
p2=svm2.predict(xval)
Printd(p2[:20])
print('MATLAB: 0 1 0 0 0 0 0 1 1 1 1 1 0 0 1 0 0 1 0 0')
