import SVM
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
np.place(test_labels, test_labels==0,-1)


print("FOR Primal")
Cs=[
100/873 ,
500/873,
700/873 ]
for c in Cs:
    print("for c = ",c)
    svm = SVM.SVM("primal")
    weights = svm.fit(features,train_labels,.01, 100,c)

    # plt.show()


    print('WEIGHTS ARE ', weights)
    # testfeatures=np.transpose(testfeatures)
    predictions= svm.predict(features)

    error_num=0
    for i,p in enumerate(predictions):
        if p != train_labels[i]:
            # print("prediction is ",p, "val is ", train_labels[i])   
            error_num+=1

    print('training error is ', error_num/len(train_labels))

    predictions= svm.predict(testfeatures)

    error_num=0
    for i,p in enumerate(predictions):
        if p != test_labels[i]:
            # print("prediction is ",p, "val is ", train_labels[i])   
            error_num+=1

    print('testing error is ', error_num/len(train_labels))

# print('FOR VOTED')

for c in Cs:
    print("for c = ",c)
    svm = SVM.SVM("dual")
    weights = svm.fit(features,train_labels,.01, 100,c)

    # plt.show()


    print('WEIGHTS ARE ', weights)
    # testfeatures=np.transpose(testfeatures)
    predictions= svm.predict(features)

    error_num=0
    for i,p in enumerate(predictions):
        if p != train_labels[i]:
            # print("prediction is ",p, "val is ", train_labels[i])   
            error_num+=1

    print('training error is ', error_num/len(train_labels))

    predictions= svm.predict(testfeatures)

    error_num=0
    for i,p in enumerate(predictions):
        if p != test_labels[i]:
            # print("prediction is ",p, "val is ", train_labels[i])   
            error_num+=1

    print('testing error is ', error_num/len(train_labels))


svm = SVM.SVM("kernel")
weights = svm.fit(features,train_labels,.01, 10)

# plt.show()


print('WEIGHTS ARE ', weights)
# testfeatures=np.transpose(testfeatures)
predictions= svm.predict(features)

error_num=0
for i,p in enumerate(predictions):
    if p != train_labels[i]:
        # print("prediction is ",p, "val is ", train_labels[i])   
        error_num+=1

print('error is ', error_num/len(train_labels))