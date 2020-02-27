import Ensemble
import numpy as np
import matplotlib.pyplot as plt
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
# features=np.transpose(features)
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
                test_lables.append(terms[i])
        data.append(feates)
        c+=1
        sample_num+=1

testfeatures=np.asarray(data)
test_labels= np.asarray(test_lables)

####training and test errors for iterations 0 >1000, training and test errors for all stumps (this will be weighted error)

print('for training data')
new_labels=np.zeros_like(train_labels,dtype=float)
for i,label in enumerate(train_labels):
    if label== 'no':
        new_labels[i]=-1
    else: 
        new_labels[i]=1


new_test_labels=np.zeros_like(test_labels,dtype=float)
for i,label in enumerate(test_labels):
    if label== 'no':
        new_test_labels[i]=-1
    else: 
        new_test_labels[i]=1
#for i in range(1,17):
# print('for',i)
train_errors=[]
test_errors=[]  
iter_counts=[]
stump_iters={}
stump_errs={}
for itr in np.arange(1, 1001):
    if itr<20 or iter ==100 or itr==200 or itr==1000 :
        ada = Ensemble.AdaBoost(itr)
        er, it =ada.fit(features,new_labels,numerical)
        stump_iters[itr]=it
        stump_errs[itr]= er
        # testfeatures=np.transpose(testfeatures)
        predictions_train= ada.predict(pred_features)
        predictions_test= ada.predict(testfeatures)

        error_num=0
        for i,p in enumerate(predictions_train):
                
            if p!=new_labels[i]:
                # print('prediction is ', p, 'actual is ', new_labels[i])
                error_num+=1
                # print('error at ', i)
        iter_counts.append(itr)
        train_errors.append(error_num/len(train_labels))
        error_num=0
        for k,p in enumerate(predictions_test):
            
            if p!=new_test_labels[k]:
                # print('prediction is ', p, 'actual is ', train_labels[i])
                error_num+=1
                # print('error at ', i)
        test_errors.append(error_num/len(test_labels))
# print(iter_counts,train_errors, test_errors)

plt.plot(iter_counts,train_errors, label='train errors ')
plt.plot(iter_counts,test_errors, label='test errors ')
plt.legend()
plt.title('Adaboost Errors')
plt.show()
plt.savefig('./adaboost_errasdf')

plt.cla()
plt.clf()

for j in stump_iters:
    # print(j)
    # print(stump_iters[j])
    # print(stump_errs[j])

    plt.plot(stump_iters[j],stump_errs[j])


plt.title('Stump Testing Errors')
plt.show()
plt.savefig('./adaboost_stump_errasdf')
        
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
        

