setwd('D:/sync/ml/ml-lib/')
source('D:/sync/ml/ml-lib/lib.r')

data=read.table('ex1data1.txt',sep=',')
n0=dim(data)[2]-1
x=as.matrix(data[,1:n0])
x=AddOnes(x)
y=data[,n0+1]

alpha=0.01
iters=1500
t0=matrix(0,2)
lamb=0

cost=LinCost(t0,x,y,lamb)
t=GradDes(t0,x,y,lamb,alpha,iters)
p1=10000*HLin(t(c(1,3.5)),t)
p2=10000*HLin(c(1,7),t)
print(c(cost,p1,p2))
# 'MATLAB: 32.0727 4519.7679 45342.4501'