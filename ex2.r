source('D:/sync/ml/ml-lib/lib.r')

data=read.table('ex2data1.txt',sep=',')
n0=dim(data)[2]-1
x=as.matrix(data[,1:n0])
x=AddOnes(x)
y=data[,n0+1]

lamb=0
t0=matrix(0,n0+1)

CostOpt=function(t){
  Cost(t,x,y,lamb)
}

cost0=CostOpt(t0)
out=optim(t0,CostOpt)
t=out$par
cost=out$value
prob=H(c(1,45,85),t)
print(c(cost0,cost,prob))
# 'MATLAB: 0.693 0.203 0.776'
