source('D:/sync/ml/ml-lib/lib.r')

data=read.table('ex1data2.txt',sep=',')
n0=dim(data)[2]-1
x=as.matrix(data[,1:n0])
x=scale(x)
x=AddOnes(x)
y=data[,n0+1]

lamb=0
alpha=0.1
iters=c(1,3,10,30,100,300)
t0=matrix(0,n0+1)

t_eqn=NormEqn(x,y)
for (i in iters){
  t=GradDes(t0,x,y,lamb,alpha,i)
  print(paste('iters=',toString(i),sep=''))
  print(cbind(t,t_eqn))
}
