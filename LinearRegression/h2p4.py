import LMS
import numpy as np
import matplotlib.pyplot as plt
import math
print('Starting p2 a, with unknowns used as attributes')
sample_num = 0
data = [] 
lable = []
with open( './data/concrete/train.csv' , 'r' ) as f:
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
print(features.shape, train_labels.shape)
# print(train_labels)

numerical=[0,5,9,11,12,13,14]
unknown=[1,3,8,15]

sample_num = 0
data = [] 
test_lables = []
with open( './data/concrete/train.csv' , 'r' ) as f:
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

print('for training data batch')

lms = LMS.LMS('batch', .5)
costs, iters, weights =lms.fit(features,train_labels,numerical)

plt.plot(iters,costs)
plt.title("Training Costs for Stochastic Gradient Descent")
# plt.show()


print('WEIGHTS ARE ', weights)
# testfeatures=np.transpose(testfeatures)
predictions= lms.predict(pred_features)


print('for training data stochastic')


lms = LMS.LMS('stochastic', .1)
costs, iters, weights =lms.fit(features,train_labels,numerical)

plt.plot(iters,costs)
plt.title("Training Costs for Stochastic Gradient Descent")
# plt.show()


print('WEIGHTS ARE ', weights)
# testfeatures=np.transpose(testfeatures)
predictions= lms.predict(pred_features)
plt.cla()

error_num=0
# print('predictions are ', predictions)

for i,p in enumerate(predictions):
    error_num+=(abs(p-train_labels[i]))

print('error is ', error_num)

print(X,Y.T)
X=X.T
res=np.linalg.inv(X*X.T)*X*Y.T

print('analytical optimal weights are ',res)