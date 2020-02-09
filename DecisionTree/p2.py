import DecisionTree
import numpy as np
print('Starting p2 a, with unknowns used as attributes')
sample_num = 0
data = [] 
lable = []
with open( './data/bank/train.csv' , 'r' ) as f:
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
pred_features= features
features=np.transpose(features)
train_labels= np.asarray(lable)
# print(features.shape)
# print(train_labels)

numerical=[0,5,9,11,12,13,14]
unknown=[1,3,8,15]

sample_num = 0
data = [] 
test_lables = []
with open( './data/bank/test.csv' , 'r' ) as f:
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
test_labels= np.asarray(lable)

print('for training data')
for split in ['entropy','gini','majority_error']:
    print('for', split)
    for i in range(1,17):
        # print('for',i)
        dt = DecisionTree.DecisionTree(split,i)
        dt.fit(features,train_labels,numerical)
        # testfeatures=np.transpose(testfeatures)
        predictions= dt.predict(pred_features)
        error_num=0
        for i,p in enumerate(predictions):
            if p!=train_labels[i]:
                # print('prediction is ', p, 'actual is ', train_labels[i])
                error_num+=1
                # print('error at ', i)

        print(error_num/len(predictions))
        
print('for testing data')
for split in ['entropy','gini','majority_error']:
    print('for', split)
    for i in range(1,17):
        # print('for',i)
        dt = DecisionTree.DecisionTree(split,i)
        dt.fit(features,train_labels,numerical)
        # testfeatures=np.transpose(testfeatures)
        predictions= dt.predict(testfeatures)
        error_num=0
        for i,p in enumerate(predictions):
            if p!=test_labels[i]:
                # print('prediction is ', p, 'actual is ', labels[i])
                error_num+=1
                # print('error at ', i)
        print(error_num/len(predictions))
        

