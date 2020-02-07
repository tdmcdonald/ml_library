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
dt = DecisionTree.DecisionTree('information_gain',12)
dt.fit(features,labels)


