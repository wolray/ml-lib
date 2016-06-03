AddOnes=function(x)
  cbind(1,x)

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

CfPara=function(xt,n){
  x=matrix(head(xt,n[1]*n[2]),n[1])
  t=matrix(tail(xt,n[2]*n[3]),n[2])
  list(x,t)
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

Grad=function(t,x,y,lamb){
  m=dim(x)[1]
  t(x)%*%(H(x,t)-y)/m+rbind(t[1]*0,as.matrix(t[-1]))*lamb/m
}

GradDes=function(t0,x,y,lamb,alpha,iters){
  t=t0
  for (i in 1:iters)
    t=t-alpha*LinGrad(t,x,y,lamb)
  t
}

H=function(x,t)
  1/(1+exp(-x%*%t))

HG=function(x)
  x*(1-x)

HLin=function(x,t)
  x%*%t

LinCost=function(t,x,y,lamb){
  m=dim(x)[1]
  sum((HLin(x,t)-y)^2)/(2*m)+sum(t[-1]^2)*lamb/2
}

LinGrad=function(t,x,y,lamb){
  m=dim(x)[1]
  t(x)%*%(HLin(x,t)-y)/m+rbind(t[1,]*0,as.matrix(t[-1,]))*lamb/m
}

NnCost=function(t,n,x,yy,lamb){
  m=dim(x)[1]
  xout=Nx(length(n),t,n,x)
  cost=sum(-yy*log(xout)-(1-yy)*log(1-xout))/m
  for (i in 1:(length(n)-1))
    cost=cost+sum(Nt(i,t,n)[-1,]^2)*lamb/(2*m)
  cost
}

NnGrad=function(t,n,x,yy,lamb){
  Hg=function(x)
    x*(1-x)
  Dt=function(k){
    if (1<k && k<length(n)){
      Dt(k+1)%*%t(Nt(k,t,n)[-1,])*Hg(Nx(k,t,n,x)[,-1])
    } else if (k==length(n)){
      Nx(k,t,n,x)-yy
    }
  }
  Gg=function(k){
    m=dim(x)[1]
    t(Nx(k,t,n,x))%*%Dt(k+1)/m+RegT(Nt(k,t,n))*lamb/m
  }
  g=c()
  for (i in 1:(length(n)-1))
    g=append(g,Gg(i))
  g
}

NnPred=function(t,n,x,y){
  m=dim(x)[1]
  xout=Nx(length(n),t,n,x)
  p=apply(xout,1,which.max)
  sum(p==y)*100/m
}

NormEqn=function(x,y)
  solve(t(x)%*%x)%*%t(x)%*%y

Nt=function(k,t,n){
  Nk=function(k)
    (n[k]+1)*n[k+1]
  Sk=function(k){
    s=0
    for (i in 1:k)
      s=s+Nk(i)
    s
  }
  if (k==1){
    matrix(t[1:Nk(k)],n[k]+1,n[k+1])
  } else if (1<k && k<length(n)){
    matrix(t[(Sk(k-1)+1):Sk(k)],n[k]+1,n[k+1])
  }
}

Nx=function(k,t,n,x){
  if (k==1){
    x
  } else if (1<k && k<length(n)){
    AddOnes(H(Nx(k-1,t,n,x),Nt(k-1,t,n)))
  } else if (k==length(n)){
    H(Nx(k-1,t,n,x),Nt(k-1,t,n))
  }
}

Ny=function(y,n){
  m=length(y)
  yy=matrix(0,m,tail(n,1))
  for (i in 1:m){
    for (j in 1:tail(n,1))
      yy[i,j]=(j==y[i])
  }
  yy
}

Pred=function(t,x,y){
  m=dim(x)[1]
  xout=H(x,t)
  p=apply(xout,1,which.max)
  sum(p==y)*100/m
}

RandT=function(n){
  s=0
  for (i in 1:(length(n)-1))
    s=s+(n[i]+1)*n[i+1]
  1-2*runif(s)
}

RegT=function(t){
  t=as.matrix(t)
  t[1,]=0
  t
}
