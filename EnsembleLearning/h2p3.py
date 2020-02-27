import Ensemble
import numpy as np
import matplotlib.pyplot as plt
import DecisionTree
import statistics
sample_num = 0
data = [] 
lable = []
numerical_cols=[0,5,9,11,12,13,14]
medians={}

arr = np.array([17,-18,56])

n=np.linalg.norm(arr)

training_feats=np.array([[1,1,-1,1,3],
[-1,1,1,2,-1],
[2,3,0,-4,-1]])
training_feats=np.transpose(training_feats)
training_labels=np.array([1,4,-1,-2,0])

# res=np.linalg.inv(X*X.T)*X*Y.T

# print('res is ', res)
weights=np.array([0,0,0])
for i in np.arange(0,5):
    new_weights=np.zeros_like(weights, dtype=float)
    
    print('iteration is ' , i)
    for j, feat in enumerate(weights):
        jw=weights[j] +.1*(training_labels[i]-weights.T.dot(training_feats[i]))*training_feats[i][j]
        print('for j', j, 'weight is ', jw)
        new_weights[j]=weights[j] +.1*(training_labels[i]-weights[:,np.newaxis].T.dot(training_feats[j]))*training_feats[i][j]
    print('and new weights is ', new_weights)
    print('and cost is ', np.linalg.norm(weights-new_weights))
    weights=new_weights
#  print(dj)
    
    # print(weights, new_weights)
   
print('array is ', arr/n)
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
train_labels= np.asarray(lable)
# print(features.shape)
# print(train_labels)
# features=np.transpose(features)
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
                test_lables.append(terms[i])
        data.append(feates)
        c+=1
        sample_num+=1
testfeatures=np.asarray(data)
test_labels= np.asarray(test_lables)

features=np.transpose(features)
testfeatures=np.transpose(testfeatures)
for i,v in enumerate(features):
    # print(training_features[i])
    if i in numerical_cols:
        # print(training_features[i].dtype)
        y=features[i].astype(np.float)
        medians[i]=statistics.median(y)
        # medians[i]=statistics.median(training_features[i])
        # print('median is ', medians[i])
print(medians)

features=np.transpose(features)
testfeatures=np.transpose(testfeatures)
if(len(numerical_cols)>0):    
    for i,v in enumerate(features):
        # print(i,features[i])
        for j,a in enumerate(features[i]):
            if j in numerical_cols:
                # print(j)
                # print(type(features[i][j]),features[i][j])
                if(float(features[i][j])>medians[j]):
                    features[i][j]='above'
                else:
                    features[i][j]='below'

if(len(numerical_cols)>0):    
    for i,v in enumerate(testfeatures):
        # print(i,features[i])
        for j,a in enumerate(testfeatures[i]):
            if j in numerical_cols:
                # print(j)
                # print(type(features[i][j]),features[i][j])
                if(float(testfeatures[i][j])>medians[j]):
                    testfeatures[i][j]='above'
                else:
                    testfeatures[i][j]='below'
numerical=[0,5,9,11,12,13,14]
unknown=[1,3,8,15]
features=np.transpose(features)
testfeatures=np.transpose(testfeatures)



print('for training data')
train_errors=[]
test_errors=[]  
iter_counts=[]

in_the_bag={}
sing_dec=DecisionTree.DecisionTree('entropy',16)
sing_dec.fit(features,train_labels)
print('done fit')
predictions_train=sing_dec.predict(pred_features)
error_num=0
for i,p in enumerate(predictions_train):
        
    if p!=train_labels[i]:
        # print('prediction is ', p, 'actual is ', new_labels[i])
        error_num+=1
        # print('error at ', i)
print(error_num/len(train_labels))

for t in np.arange(0, 100):
    in_the_bag[t]=[]

    for itr in np.arange(1,1000):
        if itr%50 == 0:
        # if True:
            bagged = Ensemble.BaggedTrees(itr)
            bagged.fit(features,train_labels,numerical)
            # testfeatures=np.transpose(testfeatures)
            # predictions_train= bagged.predict(pred_features)
            # predictions_test= bagged.predict(testfeatures)
            in_the_bag[t].append(bagged)

            # error_num=0
            # for i,p in enumerate(predictions_train):
                    
            #     if p!=train_labels[i]:
            #         # print('prediction is ', p, 'actual is ', new_labels[i])
            #         error_num+=1
            #         # print('error at ', i)
            # iter_counts.append(itr)
            # train_errors.append(error_num/len(train_labels))
            # error_num=0
            # for k,p in enumerate(predictions_test):
                
            #     if p!=test_labels[k]:
            #         # print('prediction is ', p, 'actual is ', train_labels[i])
            #         error_num+=1
            #         # print('error at ', i)
            # test_errors.append(error_num/len(test_labels))

#we now have 100*1000 god damn trees
my_first_trees={}
for t in in_the_bag:
    my_first_trees[t]=in_the_bag[t][0]

