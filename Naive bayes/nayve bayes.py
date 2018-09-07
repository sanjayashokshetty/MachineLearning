import pandas as pd
import numpy as np
data=pd.read_csv('SPECT.csv')
X=data.drop(["class"],axis=1)
Y=data["class"]
data=np.array(data)
np.random.shuffle(data)


n_features=len(data[0])-1
n_data=len(data)

for i in range(n_data):
    if(data[i][n_features]=="Yes"):
        data[i][n_features]=1
    else:
        data[i][n_features]=0

#variables
k=1
n_classes=2
k_fold=10
n_catagorial=2

def retweights():
    #atrribute category class
    prob_attr_class=[[[0 for i in range(n_classes)]for i in range(n_catagorial)] for i in range(n_features)]
    num_classes=[0 for i in range(n_classes)]
    prob_classes=[0 for i in range(n_classes)]
    return prob_attr_class,num_classes,prob_classes

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

def function(row,prob_attr_class,num_classes,prob_classes):
    y_prob=[prob_classes[i] for i in range(n_classes)]
    for i in range(n_classes):
        for j,k in enumerate(row):
            y_prob[i]*=(prob_attr_class[j][i][k]/num_classes[i])
    # print(y_prob)
    if y_prob[0]>y_prob[1]:
        return 0
    else:
        return 1

from valid import validation
foldres=validation.foldwise()

for fold in range(k_fold):
    print("fold no:",fold)
    x_train,y_train,x_test,y_test=kfolddata(fold,data)
    prob_attr_class,num_classes,prob_classes=retweights()
    foldres.reset()
    #class probability
    for i in range(len(x_train)):
        num_classes[y_train[i]]+=1
    for i in range(n_classes):
        prob_classes[i]=num_classes[i]/(len(x_train))
    #attribute probablility
    for i in range(n_features):
        for j in range(len(x_train)):
            prob_attr_class[i][y_train[j]][x_train[j][i]]=prob_attr_class[i][y_train[j]][x_train[j][i]]+1
    for i in range(len(x_test)):
        y=function(x_test[i],prob_attr_class,num_classes,prob_classes)
        z=y_test[i]
        cost=z-y
        # print(cost)
        foldres.valid(y,cost,1)
    foldres.printfoldresult()
    foldres.averageresults() 
foldres.printaverageresults()

