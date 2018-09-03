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
n_epochs=2
learning_rate=0.3
k_fold=10
eps=0.001 #epsilon to avoid divide by zero
weights=[1/(n_features+1) for i in range(n_features)]
#kfold variables
width=int(n_data/k_fold)
k_fold_index=[]
start=0
for i in range(k_fold-1):
    k_fold_index.append([start,start+width])
    start=start+width
k_fold_index.append([start,n_data])


def kfolddata(j,data):
    x_test=data[k_fold_index[j][0]:k_fold_index[j][1],0:n_features]
    y_test=data[k_fold_index[j][0]:k_fold_index[j][1],n_features]
    if(j+1<=9):
        x_train=data[k_fold_index[j][1]:,0:n_features]
        y_train=data[k_fold_index[j][1]:,n_features]
        if(j-1!=-1):
            x_train=np.append(x_train,data[0:k_fold_index[j][0],0:n_features],axis=0)
            y_train=np.append(y_train,data[0:k_fold_index[j][0],n_features],axis=0)
    else:
        x_train=data[0:k_fold_index[j][0],0:n_features]
        y_train=data[0:k_fold_index[j][0],n_features]
    return x_train,y_train,x_test,y_test

def function(row,weights,bias,thresold):
    z=0
    for i in range(len(weights)):
        z+=weights[i]*row[i]    
    z=z+bias
    if(z>thresold):
        return 1
    else:
        return 0

def update_weights(weights,row,thresold,bias,learning_rate,cost):
    for i in range(len(weights)):
        weights[i]+=learning_rate*cost*row[i]
    thresold=thresold-learning_rate*cost
    return weights,thresold,bias

from valid import validation
for epoch in range(n_epochs):
    foldres=validation.foldwise()
    for j in range(k_fold):
#         print("fold no:",j)
        foldres.reset()
        x_train,y_train,x_test,y_test=kfolddata(j,data)
        for i in range(len(x_train)):
            y=function(x_train[i],weights,bias,thresold)
            z=y_train[i]
            cost=z-y
            foldres.valid(y,cost,0)
            weights,thresold,bias=update_weights(weights,x_train[i],thresold,bias,learning_rate,cost)
        for i in range(len(x_test)):
            y=function(x_train[i],weights,bias,thresold)
            z=y_test[i]
            cost=z-y
            foldres.valid(y,cost,1)
#         foldres.printfoldresult()
        foldres.averageresults()
    if (epoch%1)==0:
        print("epoch no",epoch+1,"+"*60)
        foldres.printepochresult()   

