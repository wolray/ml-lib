from lib import *

theta_0,X,y=load_mat('ex3data1.mat')
num_labels=10
lamb=0.1

theta_all=one_all(theta_0,X,y,num_labels,lamb)
predict_one_all(theta_all,X,y)
