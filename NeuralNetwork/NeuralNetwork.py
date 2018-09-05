import pandas as pd
import numpy as np
import math
data=pd.read_csv('SPECT.csv')
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

#variables

thresold=0.5
n_epochs=1000
learning_rate=0.5
k_fold=10
layer=[n_features,5,1]
bias=[[1/layer[i+1]]*layer[i+1] for i in range(len(layer)-1)]
nodes=[[0]*layer[i] for i in range(len(layer))]
error=[[0]*layer[i] for i in range(len(layer))]

#kfold variables
width=int(n_data/k_fold)
k_fold_index=[]
start=0
for i in range(k_fold-1):
    k_fold_index.append([start,start+width])
    start=start+width
k_fold_index.append([start,n_data])

def retweights():
    #eights=[[[np.random.random_sample()]*layer[i] for j in range(layer[i+1])] for i in range(len(layer)-1)]
    weights=[[[1/(layer[i]*layer[i+1])]*layer[i] for j in range(layer[i+1])] for i in range(len(layer)-1)]
    return weights

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
    
def function(row,nodes,weights,bias,thresold,layer):
    for i in range(len(row)):
        nodes[0][i]=row[i]
    for lay in range(1,len(layer)):
        for j in range(layer[lay]):
            y=bias[lay-1][j]
            for prevlayer,k in enumerate(weights[lay-1][j]):
                y=y+nodes[lay-1][prevlayer]*k
            nodes[lay][j]=1.0/(1.0+math.exp(-y))
    if(nodes[len(layer)-1][0]>=thresold):
        y=1
    else:
        y=0
    return nodes,y

def back_prop(weights,nodes,total_error,thresold,bias,learning_rate,cost,error,z):
    #output layer error
    error[len(error)-1][0]=(z-nodes[len(error)-1][0])*(1-nodes[len(error)-1][0])*nodes[len(error)-1][0]
    #hidden layers
    total_error+=abs(error[len(error)-1][0])
    for lay in range(len(error)-2,0,-1):
        for nod in range(len(nodes[lay])):
            err=0
            for nxtlaynod in range(len(nodes[lay+1])):
                err+=error[lay+1][nxtlaynod]*weights[lay][nxtlaynod][nod]             
            error[lay][nod]=nodes[lay][nod]*(1-nodes[lay][nod])*err
    #update weights
    for lay in range(len(weights)):
        for j in range(len(weights[lay])):
            for i in range(len(weights[lay][j])):
                weights[lay][j][i]+=learning_rate*nodes[lay][i]*error[lay+1][j]
    for lay in range(len(bias)):
        for i in range(len(bias[lay])):
            bias[lay][i]+=learning_rate*error[lay+1][i]
    return weights,bias,total_error

from valid import validation
foldres=validation.foldwise()

for fold in range(k_fold):
    print("fold no:",fold)
    x_train,y_train,x_test,y_test=kfolddata(fold,data)
    weights=retweights()
    foldres.reset()
    for epoch in range(n_epochs):
        total_error=0
        for i in range(len(x_train)):
            nodes,y=function(x_train[i],nodes,weights,bias,thresold,layer)
            z=y_train[i]
            cost=z-y
            weights,bias,total_error=back_prop(weights,nodes,total_error,thresold,bias,learning_rate,cost,error,z)
#         print("epoch",epoch,"error",total_error)
    for i in range(len(x_test)):
        nodes,y=function(x_test[i],nodes,weights,bias,thresold,layer)
        z=y_test[i]
        cost=z-y
        foldres.valid(y,cost,1)
    foldres.printfoldresult()
    foldres.averageresults() 
foldres.printaverageresults()

