import Perceptron
import numpy as np
import matplotlib.pyplot as plt
import math
sample_num = 0
data = [] 
lable = []
with open( './data/bank-note/train.csv' , 'r' ) as f:
    c =0 
    for line in f:
        terms = line.strip().split(',')
        feates=[]
        for i,t in enumerate(terms):
            if i!=len(terms)-1:
                feates.append(terms[i])
            else:
                lable.append(terms[i])
        data.append(feates)
        c+=1
        sample_num+=1

features=np.asarray(data,dtype=float)

X=np.matrix(data).astype(float)
Y=np.matrix(lable).astype(float)
pred_features= features

# features=np.transpose(features)
train_labels= np.asarray(lable,dtype=float)
# print(features.shape, train_labels.shape)

sample_num = 0
data = [] 
lable = []
with open( './data/bank-note/test.csv' , 'r' ) as f:
    c =0 
    for line in f:
        terms = line.strip().split(',')
        feates=[]
        for i,t in enumerate(terms):
            if i!=len(terms)-1:
                feates.append(terms[i])
            else:
                lable.append(terms[i])
        data.append(feates)
        c+=1
        sample_num+=1

testfeatures=np.asarray(data,dtype=float)
test_labels= np.asarray(lable,dtype=float)



np.place(train_labels, train_labels==0,-1)

print("FOR STANDARD")

preceptron = Perceptron.Perceptron("margin")
weights = preceptron.fit(features,train_labels,.01, 10)

# plt.show()


print('WEIGHTS ARE ', weights)
# testfeatures=np.transpose(testfeatures)
predictions= preceptron.predict(features)

error_num=0
for i,p in enumerate(predictions):
    if p != train_labels[i]:
        # print("prediction is ",p, "val is ", train_labels[i])   
        error_num+=1

print('error is ', error_num/len(train_labels))



print('FOR VOTED')

preceptron = Perceptron.Perceptron("vote")
weights = preceptron.fit(features,train_labels,.01, 10)

# plt.show()


print('WEIGHTS ARE ', weights)
# testfeatures=np.transpose(testfeatures)
predictions= preceptron.predict(features)

error_num=0
for i,p in enumerate(predictions):
    if p != train_labels[i]:
        # print("prediction is ",p, "val is ", train_labels[i])   
        error_num+=1

print('error is ', error_num/len(train_labels))

print('FOR AVERAGE')

preceptron = Perceptron.Perceptron("average")
weights = preceptron.fit(features,train_labels,.01, 10)

# plt.show()


print('WEIGHTS ARE ', weights)
# testfeatures=np.transpose(testfeatures)
predictions= preceptron.predict(features)

error_num=0
for i,p in enumerate(predictions):
    if p != train_labels[i]:
        # print("prediction is ",p, "val is ", train_labels[i])   
        error_num+=1

print('error is ', error_num/len(train_labels))