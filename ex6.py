from lib import *

data=io.loadmat('ex6data3.mat')
X=data['X']
y=data['y'].ravel()
Xval=data['Xval']
yval=data['yval'].ravel()

clf=svm.SVC(C=1,gamma=0.01)
svm1=clf.fit(X,y)
p1=svm1.predict(Xval)
printd(p1[:20])
print('MATLAB: 1 1 0 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1')

Cs=sigmas=array([0.01,0.03,0.1,0.3,1,3,10,30])
gammas=1/(sigmas**2)
paras={'C':Cs,'gamma':gammas}
grid=grid_search.GridSearchCV(clf,paras)
svm2=grid.fit(X,y)
p2=svm2.predict(Xval)
printd(p2[:20])
print('MATLAB: 0 1 0 0 0 0 0 1 1 1 1 1 0 0 1 0 0 1 0 0')
