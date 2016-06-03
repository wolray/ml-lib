source('D:/sync/ml/ml-lib/lib.r')
library('R.matlab')

data=readMat('ex4data1.mat')
x=data$X
x=AddOnes(x)
y=data$y

data=readMat('ex4weights.mat')
t1=data$Theta1
t2=data$Theta2
t=append(t(t1),t(t2))
n=c(400,25,10)
lamb=1
yy=Ny(y,n)

cost=NnCost(t,n,x,yy,lamb)
print(cost)
## MATLAB: 0.3838

t0=RandT(n)
NnCostOpt=function(t)
  NnCost(t,n,x,yy,lamb)
NnGradOpt=function(t)
  NnGrad(t,n,x,yy,lamb)

out=optim(t0,NnCostOpt,gr=NnGradOpt,method="CG",control=list(maxit=50))
t=out$par
p=NnPred(t,n,x,y)
print(p)
## MATLAB: 95.3%
