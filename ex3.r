source('D:/sync/ml/ml-lib/lib.r')
library('R.matlab')

data=readMat('ex3data1.mat')
x=data$X
x=AddOnes(x)
y=data$y

n=c(dim(x)[2]-1,10)
t0=matrix(0,n[1]+1)
lamb=0.1
tt=matrix(0,n[1]+1,n[2])
yy=Ny(y,n)

CostOpt=function(t){
  Cost(t,x,y_temp,lamb)
}

GradOpt=function(t){
  Grad(t,x,y_temp,lamb)
}

for (k in 1:n[2]){
  y_temp=yy[,k];print(k)
  out_temp=optim(t0,CostOpt,gr=GradOpt,method="CG")
  tt[,k]=out_temp$par
}
p=Pred(tt,x,y)
print(p)
## MATLAB: 94.9%
