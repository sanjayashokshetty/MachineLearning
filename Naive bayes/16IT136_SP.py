import pandas as pd
import numpy as np
data=pd.read_csv('SPECT.csv')
X=data.drop(["class"],axis=1)
Y=data["class"]
data=np.array(data)
np.random.shuffle(data)


n_features=len(data[0])-1
n_data=len(data)

#Iris-setosa
for i in range(n_data):
    if(data[i][n_features]=="Yes"):
        data[i][n_features]=1
    else:
        data[i][n_features]=0

bias=1
thresold=0
n_epochs=10
learning_rate=0.3
k_fold=10
eps=0.001 #epsilon to avoid divide by zero
weights=[1/(n_features+1) for i in range(n_features)]

def function(row,weights,bias,thresold):
    z=0
    for i in range(len(weights)):
        z+=weights[i]*row[i]    
    z=z+bias
    if(z>thresold):
        return 1
    else:
        return 0

def valid(z,cost,a,b,c,d):
    if cost==0:
        if z==1:
            return a+1,b,c,d
        else:
            return a,b,c+1,d
    else:
        if z==0:
            return a,b+1,c,d
        else:
            return a,b,c,d+1

def update_weights(weights,row,thresold,bias,learning_rate,cost):
    for i in range(len(weights)):
        weights[i]+=learning_rate*cost*row[i]
    thresold=thresold-learning_rate*cost
    return weights,thresold,bias

def kfolddata(j,data):
    x_test=data[j*k_fold:(j+1)*k_fold,0:n_features]
    y_test=data[j*k_fold:(j+1)*k_fold,n_features]
    if(j+2<=k_fold):
        x_train=data[(j+1)*k_fold:,0:n_features]
        y_train=data[(j+1)*k_fold:,n_features]
        if(j-1>=0):
            x_train=np.append(x_train,data[0:j*k_fold,0:n_features],axis=0)
            y_train=np.append(y_train,data[0:j*k_fold,n_features],axis=0)
    else:
        x_train=data[0:j*k_fold,0:n_features]
        y_train=data[0:j*k_fold,n_features]
    return x_train,y_train,x_test,y_test

def printfoldresult(test_TT,test_TF,test_FF,test_FT,eps):
    print("trainTT:",train_TT," trainTF:",train_TF," trainFF:",train_FF," trainFT:",train_FT)
    print("train_accuracy",(train_TT+train_FF)/(train_TT+train_TF+train_FF+train_FT+eps))
    print("train Precision +:",(train_TT)/(train_TT+train_FT+eps))
    print("train Precision -:",(train_FF)/(train_FF+train_TF+eps))
    print("train recall +:",(train_TT)/(train_TT+train_TF+eps))
    print("train recall -:",(train_FF)/(train_FF+train_FT+eps))
    print("testTT:",test_TT," testTF:",test_TF," testFF:",test_FF," testFT:",test_FT)
    print("test_accuracy",(test_TT+test_FF)/(test_TT+test_TF+test_FF+test_FT+eps))
    print("test Precision +:",(test_TT)/(test_TT+test_FT+eps))
    print("test Precision -:",(test_FF)/(test_FF+test_TF+eps))
    print("test recall +:",(test_TT)/(test_TT+test_TF+eps))
    print("test recall -:",(test_FF)/(test_FF+test_FT+eps))

def averageresults(test_TT,test_TF,test_FF,test_FT,eps,accTra,accTes,precTraPos,precTraNeg,recTraPos,recTraNeg,precTesPos,precTesNeg,recTesPos,recTesNeg):
    accTra+=(train_TT+train_FF)/(train_TT+train_TF+train_FF+train_FT+eps)
    precTraPos+=((train_TT)/(train_TT+train_FT+eps))
    precTraNeg+=((train_FF)/(train_FF+train_TF+eps))
    recTraPos+=((train_TT)/(train_TT+train_TF+eps))
    recTraNeg+=((train_FF)/(train_FF+train_FT+eps))
    accTes+=((test_TT+test_FF)/(test_TT+test_TF+test_FF+test_FT+eps))
    precTesPos+=((test_TT)/(test_TT+test_FT+eps))
    precTesNeg+=((test_FF)/(test_FF+test_TF+eps))
    recTesPos+=((test_TT)/(test_TT+test_TF+eps))
    recTesNeg+=((test_FF)/(test_FF+test_FT+eps))
    return accTra,accTes,precTraPos,precTraNeg,recTraPos,recTraNeg,precTesPos,precTesNeg,recTesPos,recTesNeg 

def printepochresult(accTra,accTes,precTraPos,precTraNeg,recTraPos,recTraNeg,precTesPos,precTesNeg,recTesPos,recTesNeg,k_fold):
    print("average result for ",k_fold,"fold")
    print("train_accuracy:",accTra/k_fold)
    print("train Precision +:",precTraPos/k_fold)
    print("train Precision -:",precTraNeg/k_fold)
    print("train recall +:",recTraPos/k_fold)
    print("train recall -:",recTraNeg/k_fold)
    print("test_accuracy",accTes/k_fold)
    print("test Precision +:",precTesPos/k_fold)
    print("test Precision -:",precTesNeg/k_fold)
    print("test recall +:",recTesPos/k_fold)
    print("test recall -:",recTesNeg/k_fold)

for i in range(n_epochs):
    print("epoch no",i+1,"+"*60)
    accTra,accTes,precTraPos,precTraNeg,recTraPos,recTraNeg,precTesPos,precTesNeg,recTesPos,recTesNeg=(0 for i in range(10))
    for j in range(k_fold):
        # print("fold no:",j)
        x_train,y_train,x_test,y_test=kfolddata(j,data)
        #validation variables
        train_TT,train_TF,train_FF,train_FT,test_TT,test_TF,test_FF,test_FT=(0 for i in range(8))
        for i in range(len(x_train)):
            y=function(x_train[i],weights,bias,thresold)
            z=y_train[i]
            cost=z-y
            train_TT,train_TF,train_FF,train_FT=valid(y,cost,train_TT,train_TF,train_FF,train_FT)
            weights,thresold,bias=update_weights(weights,x_train[i],thresold,bias,learning_rate,cost)
        for i in range(len(x_test)):
            y=function(x_test[i],weights,bias,thresold)
            z=y_test[i]
            cost=z-y
            test_TT,test_TF,test_FF,test_FT=valid(y,cost,test_TT,test_TF,test_FF,test_FT)
        # printfoldresult(test_TT,test_TF,test_FF,test_FT,eps)
        accTra,accTes,precTraPos,precTraNeg,recTraPos,recTraNeg,precTesPos,precTesNeg,recTesPos,recTesNeg=averageresults(test_TT,test_TF,test_FF,test_FT,eps,accTra,accTes,precTraPos,precTraNeg,recTraPos,recTraNeg,precTesPos,precTesNeg,recTesPos,recTesNeg)
    printepochresult(accTra,accTes,precTraPos,precTraNeg,recTraPos,recTraNeg,precTesPos,precTesNeg,recTesPos,recTesNeg,k_fold)
