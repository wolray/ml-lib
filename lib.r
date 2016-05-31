AddOnes=function(x){
  cbind(1,x)
}

CfPara=function(xt,n){
  x=matrix(head(xt,n[1]*n[2]),n[1])
  t=matrix(tail(xt,n[2]*n[3]),n[2])
  list(x,t)
}

CfCost=function(xt,n,y,r,lamb){
  x=CfPara(xt,n)[[1]]
  t=CfPara(xt,n)[[2]]
  sum(((HLin(x,t)-y)*r)^2)/2+(sum(x^2)+sum(t^2))*lamb/2
}

CfGrad=function(xt,n,y,r,lamb){
  x=CfPara(xt,n)[[1]]
  t=CfPara(xt,n)[[2]]
  x_grad=(HLin(x,t)-y)*r%*%t(t)+x*lamb
  t_grad=t((HLin(x,t)-y)*r)%*%x+t*lamb
  append(x_grad,t_grad)
}

Cost=function(t,x,y,lamb){
  m=dim(x)[1]
  sum(-y*log(H(x,t))-(1-y)*log(1-H(x,t)))/m+sum(t[-1]^2)*lamb/(2*m)
}

Epsilon=function(yval,pval){
  f1_best=eps_best=0
  for (eps in seq(min(pval),max(pval),length.out=1000))
    tp=sum((yval==1)*(pval<eps))
    fp=sum((yval==0)*(pval<eps))
    fn=sum((yval==1)*(pval>=eps))
    prec=tp/(tp+fp)
    rec=tp/(tp+fn)
    f1=2*prec*rec/(prec+rec)
    if (f1>f1_best){
      f1_best=f1
      eps_best=eps
    }
    list(eps_best,f1_best)
}

H=function(x,t){
  1/(1+exp(-x%*%t))
}

HLin=function(x,t){
  x%*%t
}

LinCost=function(t,x,y,lamb){
  m=dim(x)[1]
  sum((HLin(x,t)-y)^2)/(2*m)+sum(t[-1]^2)*lamb/2
}

LinGrad=function(t,x,y,lamb){
  m=dim(x)[1]
  t(x)%*%(HLin(x,t)-y)/m+rbind(t[1,]*0,as.matrix(t[-1,]))*lamb/m
}

GradDes=function(t0,x,y,lamb,alpha,iters){
  t=t0
  for (i in 1:iters){
    t=t-alpha*LinGrad(t,x,y,lamb)
  }
  t
}

NormEqn=function(x,y){
  solve(t(x)%*%x)%*%t(x)%*%y
}

Ny=function(y,n){
  m=length(y)
  yy=matrix(0,m,tail(n,1))
  for (i in 1:m){
    for (j in 1:tail(n,1)){
      yy[i,j]=(j==y[i])
    }
  }
  yy
}

Grad=function(t,x,y,lamb){
  m=dim(x)[1]
  t(x)%*%(H(x,t)-y)/m+rbind(t[1]*0,as.matrix(t[-1]))*lamb/m
}

Pred=function(t,x,y){
  m=dim(x)[1]
  xout=H(x,t)
  p=apply(xout,1,which.max)
  sum(p==y)*100/m
}