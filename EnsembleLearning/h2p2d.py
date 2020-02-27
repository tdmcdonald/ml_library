import Ensemble
import numpy as np
import matplotlib.pyplot as plt
import statistics
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

numerical_cols=[0,5,9,11,12,13,14]
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


    # print('for',i)
plot_dick={}
for j in [2,4,6]:
    train_errors=[]
    test_errors=[]  
    iter_counts=[]
    for i in range(1,1001):
        if(i<20 or i == 200 or i==1000):
            print('starting iter ', i, j)
            rf = Ensemble.RandomForrest(i,j)
            rf.fit(features,train_labels)
            predictions_train= rf.predict(pred_features)
            predictions_test= rf.predict(testfeatures)

            error_num=0
            for k,p in enumerate(predictions_train):
                
                if p!=train_labels[k]:
                    # print('prediction is ', p, 'actual is ', train_labels[i])
                    error_num+=1
                    # print('error at ', i)
            iter_counts.append(i)
            train_errors.append(error_num/len(train_labels))
            error_num=0
            for k,p in enumerate(predictions_test):
                
                if p!=test_labels[k]:
                    # print('prediction is ', p, 'actual is ', train_labels[i])
                    error_num+=1
                    # print('error at ', i)
            test_errors.append(error_num/len(train_labels))
        plot_dick[j]=[iter_counts,test_errors]
        plt.plot(iter_counts,train_errors, label='sample size '+str(j))
        plt.legend()
        plt.title('Random Forrest Training Errors')
        # plt.show()
        plt.savefig('./rand_for_train_new')

        plt.clf()
        plt.cla()
        plt.plot(iter_counts,test_errors, label='sample size '+str(j))
        plt.legend()
        plt.title('Random Forrest Training Errors')
        plt.savefig('./rand_for_test_new')


# plt.legend()
# plt.title('Random Forrest Training Errors')
# # plt.show()
# plt.savefig('./rand_for_train_new')
# plt.clf()
# plt.cla()
# for j in plot_dick:
#     print(j)
#     plt.plot(plot_dick[j][0],plot_dick[j][1],label='sample size '+str(j))

# plt.legend()
# plt.title('Random Forrest Testing Errors')
# plt.savefig('./rand_for_test_new')

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
        

