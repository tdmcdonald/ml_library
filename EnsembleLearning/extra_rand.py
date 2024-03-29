import Ensemble
import numpy as np
import matplotlib.pyplot as plt
import statistics
sample_num = 0
data = [] 
lable = []
with open( './data/credit_card/credit_card.csv' , 'r' ) as f:
    c =0 
    for line in f:
        terms = line.strip().split(',')
        feates=[]
        for i,t in enumerate(terms):
            if i!=len(terms)-1:
                feates.append(terms[i])
            elif i>1:
                lable.append(terms[i])
        data.append(feates)
        c+=1
        sample_num+=1

all_features=np.asarray(data)
print(all_features.shape)
features=all_features[0:24000,:]

pred_features= features
# features=np.transpose(features)
train_labels= np.asarray(lable)[0:24000]
testfeatures=all_features[24000:3000,:]
test_labels=np.asarray(lable)[24000:3000]
print(features)
print(testfeatures.shape)
print(train_labels.shape)
print(test_labels.shape)
# print(train_labels)

numerical_cols=np.arange(5,23)

sample_num = 0
data = [] 
test_lables = []


testfeatures=np.asarray(data)
test_labels= np.asarray(test_lables)
medians={}
features=np.transpose(features)
for i,v in enumerate(features):
    # print(training_features[i])
    if i in numerical_cols:
        # print(training_features[i].dtype)
        y=features[i].astype(np.float)
        medians[i]=statistics.median(y)
        # medians[i]=statistics.median(training_features[i])
        # print('median is ', medians[i])
features=np.transpose(features)

if(len(numerical_cols)>0):    
    for i,v in enumerate(features):
        # print(i,features[i])
        for j,a in enumerate(features[i]):
            if j in numerical_cols:
                # print(j)
                # print(type(features[i][j]),features[i][j])
                if(float(features[i][j])>=medians[j]):
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
                if(float(testfeatures[i][j])>=medians[j]):
                    testfeatures[i][j]='above'
                else:
                    testfeatures[i][j]='below'

print('for training data')
train_errors=[]
test_errors=[]  
iter_counts=[]

for itr in np.arange(1,1000):
    if itr<20 or itr%200 == 0:
        print('starting for iter ', itr)
    # if True:
        bagged = Ensemble.RandomForrest(itr, 4)
        bagged.fit(features,train_labels)
        # testfeatures=np.transpose(testfeatures)
        predictions_train= bagged.predict(pred_features)
        predictions_test= bagged.predict(testfeatures)

        error_num=0
        for i,p in enumerate(predictions_train):
                
            if p!=train_labels[i]:
                # print('prediction is ', p, 'actual is ', new_labels[i])
                error_num+=1
                # print('error at ', i)
        iter_counts.append(itr)
        print('train err ',(error_num/len(train_labels)))

        train_errors.append(error_num/len(train_labels))
        error_num=0
        for k,p in enumerate(predictions_test):
            
            if p!=test_labels[k]:
                # print('prediction is ', p, 'actual is ', train_labels[i])
                error_num+=1
                # print('error at ', i)
        print('test err ',(error_num/len(test_labels)))
        test_errors.append(error_num/len(test_labels))
        plt.clf()
        plt.cla()       
        plt.plot(iter_counts,train_errors, label='train errors ')
        plt.plot(iter_counts,test_errors, label='test errors ')
        plt.legend()
        plt.title('Random Forrest Errors') 
        plt.savefig('./rand_extra')
print(iter_counts,train_errors, test_errors)
plt.clf()
plt.cla()       
plt.plot(iter_counts,train_errors, label='train errors ')
plt.plot(iter_counts,test_errors, label='test errors ')
plt.legend()
plt.title('Random Forrest Errors') 
plt.savefig('./rand_extra')
#for i in range(1,17):
# print('for',i)

        
# print('for testing data')

#     #for i in range(1,17):
#         # print('for',i)
# dt = DecisionTree.DecisionTree('entropy',50)
# dt.fit(features,train_labels,numerical)
# # testfeatures=np.transpose(testfeatures)
# predictions= dt.predict(testfeatures)
# error_num=0
# for i,p in enumerate(predictions):
#     if p!=test_labels[i]:
#         # print('prediction is ', p, 'actual is ', labels[i])
#         error_num+=1
#         # print('error at ', i)
# print(error_num/len(predictions))
        

