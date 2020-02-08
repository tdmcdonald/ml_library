import DecisionTree
import numpy as np
print('main')
sample_num = 0
data = [] 
lable = []
with open( './data/car/train.csv' , 'r' ) as f:
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

features=np.asarray(data)
features=np.transpose(features)
labels= np.asarray(lable)
print(features.shape)
dt = DecisionTree.DecisionTree('majority_error',12)
dt.fit(features,labels)

sample_num = 0
data = [] 
lable = []
with open( './data/car/test.csv' , 'r' ) as f:
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

testfeatures=np.asarray(data)
testfeatures=np.transpose(testfeatures)
predictions= dt.predict(testfeatures)
error_num=0
for i,p in enumerate(predictions):
    # print('prediction is ', p, 'actual is ', labels[i])
    if p!=labels[i]:
        error_num+=1

print('total incorrect predictions is: ', error_num)
