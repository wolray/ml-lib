from lib_ln import *

data=io.loadmat('ex5data1.mat')
X=data['X']
m,n0=X.shape
X=add_ones(X)
y=data['y']
Xval=data['Xval']
yval=data['yval']
Xval=add_ones(Xval)

t0=zeros((n0+1,1))
lamb=0
alpha=0.001
iters=5000

tcg=cfmin_cg(t0,X,y,lamb)
tmin=cfmin(t0,X,y,lamb)[0]
tdes=grad_des(zeros((2,1)),X,y,lamb,alpha,iters)
J=cost(t0,X,y,lamb)
Jcg=cost(tcg,X,y,lamb)
Jmin=cost(tmin,X,y,lamb)
Jdes=cost(tdes,X,y,lamb)

print()
print(t0.flatten(),tcg,tmin,tdes)
print(J,Jcg,Jmin,Jdes)
print('ex5: 303.993192')

g=grad(t0,X,y,lamb)

print()
print(g.flatten())
print('ex5: [-15.303016 598.250744]')

lamb=0
def curve(t0,X,y,Xval,yval,lamb):
    errt=zeros(m)
    errv=zeros(m)
    for i in range(1,m+1):
        t=grad_des(t0,X[:i],y[:i],lamb,alpha,iters)
        errt[i-1]=cost(t,X[:i],y[:i],0)
        errv[i-1]=cost(t,Xval,yval,0)
    return errt,errv

lamb=0
errt,errv=curve(t0,X,y,Xval,yval,lamb)

print()
print(errt)
print(errv)

def poly(p,x):
    m=x.shape[0]
    xp=zeros((m,p))
    for i in range(p):
        xp[:,i]=(x[:,-1]**(i+1))
    return norm_features(xp)

p=8
Xp=poly(p,X)
Xvalp=poly(p,Xval)
t0=zeros((p+1,1))
errt,errv=curve(t0,Xp,y,Xvalp,yval,lamb)

print()
print(errt)
print(errv)
