source('D:/sync/ml/ml-lib/lib.r')

data=read.table('ex2data2.txt',sep=',')
n0=dim(data)[2]-1
x=as.matrix(data[,1:n0])
x=AddOnes(x)
y=data[,n0+1]

lamb_group=c(0,1,10,100)
t0=matrix(0,n0+1)

CostOpt=function(t){
  Cost(t,x,y,lamb)
}

cost0=Cost(t0,x,y,0)
for (lamb in lamb_group){
  out=optim(t0,CostOpt)
  t=out$par
  cost=out$value
  print(paste('lamb=',toString(lamb),sep=''))
  print(c(cost0,cost))
}
