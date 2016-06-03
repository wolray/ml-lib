source('D:/sync/ml/ml-lib/lib.r')
library('R.matlab')

data=readMat('ex3data1.mat')
x=data$X
x=AddOnes(x)
y=data$y

data=readMat('ex3weights.mat')
t1=data$Theta1
t2=data$Theta2
t=append(t(t1),t(t2))
n=c(400,25,10)
lamb=0

p=NnPred(t,n,x,y)
print(p)
## MATLAB: 97.5%
